"""Minimal supervised trainer with class weighting and early stopping on validation macro-F1."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_proba(
    model: nn.Module, X: np.ndarray, batch_size: int = 128, device: str = "cpu"
) -> np.ndarray:
    model.eval().to(device)
    out = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size):
        out.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: TrainConfig,
    seed: int = 42,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train and return the model state with the best validation macro-F1, plus history."""
    torch.manual_seed(seed)
    model.to(cfg.device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, num_classes).to(cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        scheduler.step()

        val_pred = predict_proba(model, X_val, device=cfg.device).argmax(axis=1)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_state, stale = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history
