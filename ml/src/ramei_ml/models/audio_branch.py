"""Audio branch (TRD §4.1): WavLM-Base+ -> layer-weighted sum -> attentive stats pooling
-> z_a (768-d) -> emotion logits.

Fine-tuning modes:
- ``frozen``: encoder frozen; trains layer weights, pooling, projection and head (linear probe+).
- ``lora``:   LoRA on attention projections (TRD: r=8, alpha=16) + the trainable parts above.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel

from ramei_ml.models.pooling import AttentiveStatsPooling, lengths_to_mask


@dataclass
class AudioBranchConfig:
    pretrained: str | None = "microsoft/wavlm-base-plus"
    # used only when pretrained is None: build a randomly initialized encoder (tests, debugging)
    encoder_config: dict[str, Any] | None = None
    mode: str = "lora"  # "frozen" | "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    embed_dim: int = 768
    head_hidden: int = 256
    dropout: float = 0.2
    num_classes: int = 7
    gradient_checkpointing: bool = False


class AudioBranch(nn.Module):
    def __init__(self, cfg: AudioBranchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # LayerDrop must be off: the layer-weighted sum needs every hidden state on every step
        if cfg.pretrained:
            encoder = WavLMModel.from_pretrained(cfg.pretrained, layerdrop=0.0)
        else:
            encoder = WavLMModel(WavLMConfig(**{**(cfg.encoder_config or {}), "layerdrop": 0.0}))
        encoder.freeze_feature_encoder()  # CNN front-end is never fine-tuned
        hidden = encoder.config.hidden_size
        n_layers = encoder.config.num_hidden_layers + 1  # + embedding output

        if cfg.mode == "frozen":
            encoder.requires_grad_(False)
        elif cfg.mode == "lora":
            from peft import LoraConfig, inject_adapter_in_model

            encoder.requires_grad_(False)
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_targets),
            )
            encoder = inject_adapter_in_model(lora, encoder)  # LoRA params are trainable
        else:
            raise ValueError(f"Unknown mode '{cfg.mode}'")
        if cfg.gradient_checkpointing:
            encoder.gradient_checkpointing_enable()
        self.encoder = encoder

        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.layer_norm = nn.LayerNorm(hidden)
        self.pool = AttentiveStatsPooling(hidden)
        self.project = nn.Sequential(
            nn.Linear(2 * hidden, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.num_classes),
        )

    def frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder._get_feat_extract_output_lengths(sample_lengths).long()

    def forward(self, waves: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """waves [B, S] zero-padded 16 kHz audio, lengths [B] in samples."""
        frozen = self.cfg.mode == "frozen"
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            # group-norm WavLM: zero-pad and do not pass an attention mask (HF guidance)
            out = self.encoder(waves, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)  # [L, B, T, H]
        weights = torch.softmax(self.layer_weights, dim=0).to(hidden.dtype)
        x = self.layer_norm((weights[:, None, None, None] * hidden).sum(dim=0))  # [B, T, H]

        frames = self.frame_lengths(lengths).clamp(min=1, max=x.size(1))
        mask = lengths_to_mask(frames, x.size(1))
        z = self.project(self.pool(x, mask))
        return {"z": z, "logits": self.head(z), "layer_weights": weights.detach()}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Only trainable parameters (LoRA + head) — a few MB instead of the full encoder."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def parameter_groups(self, lr_encoder: float, lr_head: float) -> list[dict]:
        enc = [
            p for n, p in self.named_parameters() if p.requires_grad and n.startswith("encoder.")
        ]
        rest = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("encoder.")
        ]
        groups = [{"params": rest, "lr": lr_head}]
        if enc:
            groups.append({"params": enc, "lr": lr_encoder})
        return groups
