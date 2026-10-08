"""Evaluation metrics shared by every experiment (TRD §6.1)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _clip_renorm(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Avoid log(0) while keeping rows valid distributions."""
    clipped = np.clip(probs, eps, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_report(
    y_true: np.ndarray, probs: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """WA, UA, macro-F1, per-class F1, calibration metrics and confusion matrix."""
    y_pred = probs.argmax(axis=1)
    class_ids = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    present = np.isin(class_ids, np.unique(y_true))
    return {
        "n": int(len(y_true)),
        "weighted_accuracy": float(accuracy_score(y_true, y_pred)),
        "unweighted_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # macro-F1 over classes that appear in y_true (absent classes would only add zeros)
        "macro_f1": float(per_class[present].mean()),
        "per_class_f1": {
            lab: float(f) for lab, f, p in zip(labels, per_class, present, strict=True) if p
        },
        "ece": expected_calibration_error(probs, y_true),
        "nll": float(log_loss(y_true, _clip_renorm(probs), labels=class_ids)),
        "brier": brier_score(probs, y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
    }


def group_report(
    y_true: np.ndarray, probs: np.ndarray, groups: pd.Series, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Macro-F1 / UA per group (e.g. gender, dataset) for fairness reporting (NFR-09)."""
    out: dict[str, dict[str, float]] = {}
    groups = groups.fillna("unknown").to_numpy()
    for g in sorted(set(groups)):
        mask = groups == g
        rep = classification_report(y_true[mask], probs[mask], labels)
        out[str(g)] = {
            "n": rep["n"],
            "macro_f1": rep["macro_f1"],
            "unweighted_accuracy": rep["unweighted_accuracy"],
        }
    return out
