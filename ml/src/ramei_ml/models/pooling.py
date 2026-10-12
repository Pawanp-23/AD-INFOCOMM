"""Masked pooling layers over frame sequences."""

import torch
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean mask [B, T], True for valid frames."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling: attention-weighted mean and std over valid frames.

    Input  x [B, T, D], mask [B, T]  ->  output [B, 2D]
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores.float(), dim=1).to(x.dtype).unsqueeze(-1)  # [B, T, 1]
        mean = (w * x).sum(dim=1)
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp_min(1e-6))
        return torch.cat([mean, std], dim=-1)
