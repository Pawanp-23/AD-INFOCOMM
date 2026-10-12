"""RAMEI fusion model on frozen-encoder features (TRD §4.4-4.6, ADR-0001).

    audio  [13, 1536] --AudioHead--> z_a, logits_a ┐
    text   [768]+[7]  --TextHead---> z_t, logits_t ├─> REM_m(quality_m, uncertainty_m) -> r_m
    visual [16, 768]  --VisualHead-> z_v, logits_v ┘
    [CLS, z_a, z_t, z_v] --Reliability-Gated Cross-Modal Transformer--> fused logits

Attention to modality token k gets an additive bias beta * log(r_k + eps), and -inf when the
modality is missing. r is detached: reliability is learned only from its own supervision
(was this branch right?), so it stays an interpretable probability instead of a free gate.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MODALITIES = ("audio", "text", "visual")


@dataclass
class FusionConfig:
    hidden: int = 256  # branch embedding size
    d_model: int = 128  # fusion transformer width
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    num_classes: int = 7
    audio_layers: int = 13
    audio_dim: int = 1536
    text_dim: int = 768
    visual_dim: int = 768
    frames: int = 16
    quality_dims: tuple[int, int, int] = (5, 5, 6)  # audio, text, visual
    use_reliability: bool = True  # False = ablation E9 (masking only, no reliability bias)


def uncertainty_features(logits: torch.Tensor) -> torch.Tensor:
    """[entropy (normalized), max prob, top-2 margin] from branch logits."""
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-9).log()).sum(-1) / torch.log(torch.tensor(p.size(-1)))
    top2 = p.topk(2, dim=-1).values
    return torch.stack([entropy, top2[..., 0], top2[..., 0] - top2[..., 1]], dim=-1)


class AudioHead(nn.Module):
    def __init__(self, c: FusionConfig) -> None:
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(c.audio_layers))
        self.net = nn.Sequential(
            nn.LayerNorm(c.audio_dim),
            nn.Linear(c.audio_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.out = nn.Linear(c.hidden, c.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.layer_weights.softmax(0)
        z = self.net((w[None, :, None] * x).sum(1))
        return z, self.out(z)


class TextHead(nn.Module):
    """Uses the MELD-trained encoder's embedding and its emotion logits as inputs."""

    def __init__(self, c: FusionConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c.text_dim + c.num_classes),
            nn.Linear(c.text_dim + c.num_classes, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.out = nn.Linear(c.hidden, c.num_classes)

    def forward(
        self, z_text: torch.Tensor, prior_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net(torch.cat([z_text, prior_logits], dim=-1))
        return z, self.out(z)


class VisualHead(nn.Module):
    """Temporal transformer over per-frame face embeddings with a [CLS] token."""

    def __init__(self, c: FusionConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(c.visual_dim), nn.Linear(c.visual_dim, c.hidden))
        self.cls = nn.Parameter(torch.zeros(1, 1, c.hidden))
        self.pos = nn.Parameter(torch.zeros(1, c.frames + 1, c.hidden))
        layer = nn.TransformerEncoderLayer(
            c.hidden, 4, 2 * c.hidden, c.dropout, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Sequential(nn.Dropout(c.dropout), nn.Linear(c.hidden, c.num_classes))

    def forward(
        self, frames: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = frames.size(0)
        x = torch.cat([self.cls.expand(b, -1, -1), self.proj(frames)], dim=1) + self.pos
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.temporal(x, src_key_padding_mask=~valid)[:, 0]
        return z, self.out(z)


class ReliabilityEstimator(nn.Module):
    """r_m = sigmoid(MLP([normalized quality_m ; uncertainty(logits_m)]))."""

    def __init__(self, quality_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(quality_dim + 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, quality: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        x = torch.cat([quality, uncertainty_features(logits.detach())], dim=-1)
        return self.net(x).squeeze(-1)  # logit of r_m


class GatedFusionLayer(nn.Module):
    """Pre-norm transformer layer whose attention takes an additive per-key bias."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        mask = key_bias[:, None, :].expand(b, n, n).repeat_interleave(self.heads, dim=0)
        h = self.norm1(x)
        out, weights = self.attn(
            h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True
        )
        x = x + self.drop(out)
        return x + self.drop(self.ff(self.norm2(x))), weights


class RAMEIFusion(nn.Module):
    def __init__(self, c: FusionConfig = FusionConfig()) -> None:
        super().__init__()
        self.c = c
        self.audio, self.text, self.visual = AudioHead(c), TextHead(c), VisualHead(c)
        self.rem = nn.ModuleDict(
            {m: ReliabilityEstimator(q) for m, q in zip(MODALITIES, c.quality_dims, strict=True)}
        )
        self.token_proj = nn.ModuleDict({m: nn.Linear(c.hidden, c.d_model) for m in MODALITIES})
        self.modality_embed = nn.Parameter(torch.zeros(len(MODALITIES), c.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.layers = nn.ModuleList(
            GatedFusionLayer(c.d_model, c.heads, c.dropout) for _ in range(c.layers)
        )
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch keys: audio, text_z, text_logits, visual, visual_mask, q_audio, q_text,
        q_visual, avail [B, 3] (bool, modality present)."""
        z_a, l_a = self.audio(batch["audio"])
        z_t, l_t = self.text(batch["text_z"], batch["text_logits"])
        z_v, l_v = self.visual(batch["visual"], batch["visual_mask"])
        zs, branch_logits = (z_a, z_t, z_v), (l_a, l_t, l_v)
        rel_logits = torch.stack(
            [
                self.rem[m](batch[f"q_{m}"], lg)
                for m, lg in zip(MODALITIES, branch_logits, strict=True)
            ],
            1,
        )
        avail = batch["avail"]
        reliability = torch.sigmoid(rel_logits) * avail  # missing modality -> r = 0

        tokens = (
            torch.stack([self.token_proj[m](z) for m, z in zip(MODALITIES, zs, strict=True)], dim=1)
            + self.modality_embed
        )
        x = torch.cat([self.cls.expand(tokens.size(0), -1, -1), tokens], dim=1)
        bias = torch.zeros_like(reliability)
        if self.c.use_reliability:
            bias = self.beta * torch.log(reliability.detach() + 1e-4)
        bias = bias.masked_fill(~avail.bool(), float("-inf"))
        key_bias = torch.cat([torch.zeros_like(bias[:, :1]), bias], dim=1)  # CLS key: no bias
        attn = None
        for layer in self.layers:
            x, attn = layer(x, key_bias)
        weights = attn[:, 0, 1:]  # CLS -> modality attention in the last layer
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return {
            "logits": self.head(self.norm(x[:, 0])),
            "branch_logits": torch.stack(branch_logits, dim=1),  # [B, 3, C]
            "reliability_logits": rel_logits,  # [B, 3]
            "reliability": reliability,  # [B, 3]
            "weights": weights,  # [B, 3] effective fusion weights
        }


def fusion_loss(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    avail: torch.Tensor,
    class_weight: torch.Tensor | None,
    aux: float = 0.3,
    rel: float = 0.5,
    label_smoothing: float = 0.1,
) -> dict[str, torch.Tensor]:
    """L = CE(fused) + aux * CE(available branches) + rel * BCE(r_m, branch_m correct)."""
    fused = F.cross_entropy(out["logits"], y, weight=class_weight, label_smoothing=label_smoothing)
    bl = out["branch_logits"]
    m = avail.float()
    ce_b = F.cross_entropy(
        bl.flatten(0, 1), y.repeat_interleave(bl.size(1)), weight=class_weight, reduction="none"
    ).view_as(m)
    branch = (ce_b * m).sum() / m.sum().clamp_min(1)
    correct = (bl.argmax(-1) == y[:, None]).float()
    bce = F.binary_cross_entropy_with_logits(out["reliability_logits"], correct, reduction="none")
    reliability = (bce * m).sum() / m.sum().clamp_min(1)
    total = fused + aux * branch + rel * reliability
    return {"total": total, "fused": fused, "branch": branch, "reliability": reliability}
