"""M0 baseline models.

- Thesis replica (E0): StandardScaler + sklearn MLPClassifier on utterance statistics.
  Hyperparameters match the MLPClassifier setup the thesis describes (300 hidden units,
  alpha=0.01, batch 256, adaptive learning rate, 500 iterations).
- Repo replica (E1): 1D-CNN over MFCC+Δ+ΔΔ time series, in the spirit of
  MiteshPuthran/Speech-Emotion-Analyzer, with BatchNorm and global pooling added.
"""

from collections.abc import Sequence

import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def build_mlp(
    hidden_layer_sizes: Sequence[int] = (300,),
    alpha: float = 0.01,
    batch_size: int = 256,
    learning_rate: str = "adaptive",
    max_iter: int = 500,
    seed: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


class CNN1D(nn.Module):
    """Input (B, C, T) -> logits (B, num_classes)."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        def block(c_in: int, c_out: int, pool: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool) if pool > 1 else nn.Identity(),
            )

        self.features = nn.Sequential(
            block(in_channels, 256, 1),
            block(256, 128, 4),
            nn.Dropout(dropout),
            block(128, 128, 2),
            block(128, 128, 1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * 128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)  # global avg + max pooling
        return self.head(pooled)
