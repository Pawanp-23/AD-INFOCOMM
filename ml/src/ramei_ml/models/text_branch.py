"""Text branch (TRD §4.2): DeBERTa-v3 encoder + LoRA -> [CLS] -> z_t (768-d) -> emotion logits."""

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class TextBranchConfig:
    pretrained: str = "microsoft/deberta-v3-base"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: list[str] = field(default_factory=lambda: ["query_proj", "value_proj"])
    dropout: float = 0.1
    num_classes: int = 7
    max_length: int = 128


class TextBranch(nn.Module):
    def __init__(self, cfg: TextBranchConfig) -> None:
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        self.cfg = cfg
        encoder = AutoModel.from_pretrained(cfg.pretrained, dtype=torch.float32)
        encoder.requires_grad_(False)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets),
        )
        self.encoder = inject_adapter_in_model(lora, encoder)
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(hidden, cfg.num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        z = out.last_hidden_state[:, 0]  # [CLS]
        return {"z": z, "logits": self.head(self.dropout(z))}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}
