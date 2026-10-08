"""Stage S0 — baseline experiments E0 (thesis MLP) and E1 (repo 1D-CNN). TRD §5, §6.2.

Run from ml/:
    python -m ramei_ml.training.baseline                                  # E0 on RAVDESS
    python -m ramei_ml.training.baseline experiment=e1_cnn
    python -m ramei_ml.training.baseline data=ravdess_random              # leakage demonstration

Each run writes to artifacts/experiments/<experiment>__<data>/<timestamp>/ and appends a row
to artifacts/experiments/leaderboard.csv.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ramei_ml.data.labels import EMOTION_TO_ID, EMOTIONS, NUM_CLASSES
from ramei_ml.data.manifest import assert_speaker_independent
from ramei_ml.evaluation.metrics import classification_report, group_report
from ramei_ml.features.mfcc import extract_manifest_features
from ramei_ml.models.baseline import CNN1D, build_mlp
from ramei_ml.paths import ARTIFACTS_DIR, PROCESSED_DIR, resolve
from ramei_ml.training.torch_trainer import TrainConfig, fit, predict_proba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = list(EMOTIONS)


def _split_arrays(df: pd.DataFrame, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    y = df["emotion"].map(EMOTION_TO_ID).to_numpy()
    out = {}
    for split in ("train", "val", "test"):
        mask = (df["split"] == split).to_numpy()
        out[split] = (X[mask], y[mask])
    return out


def _run_mlp(
    cfg: DictConfig, splits: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    X_tr, y_tr = splits["train"]
    model = build_mlp(seed=cfg.seed, **OmegaConf.to_container(cfg.experiment.mlp))
    model.fit(X_tr, y_tr)

    def full_proba(X: np.ndarray) -> np.ndarray:  # classes absent from train get probability 0
        proba = np.zeros((len(X), NUM_CLASSES), dtype=np.float64)
        proba[:, model.classes_] = model.predict_proba(X)
        return proba

    probs = {s: full_proba(splits[s][0]) for s in ("val", "test")}
    return model, probs, {"n_iter": int(model[-1].n_iter_)}


def _run_cnn(
    cfg: DictConfig, splits: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    X_tr, y_tr = splits["train"]
    mean = X_tr.mean(axis=(0, 2), keepdims=True)  # per-channel normalization fitted on train only
    std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = {s: ((x - mean) / std).astype(np.float32) for s, (x, _) in splits.items()}

    c = cfg.experiment.cnn
    model = CNN1D(in_channels=X_tr.shape[1], num_classes=NUM_CLASSES, dropout=c.dropout)
    tcfg = TrainConfig(
        epochs=c.epochs,
        batch_size=c.batch_size,
        lr=c.lr,
        weight_decay=c.weight_decay,
        patience=c.patience,
    )
    model, history = fit(
        model, norm["train"], y_tr, norm["val"], splits["val"][1], NUM_CLASSES, tcfg, seed=cfg.seed
    )
    probs = {s: predict_proba(model, norm[s]) for s in ("val", "test")}
    bundle = {"state_dict": model.state_dict(), "mean": mean, "std": std}
    return bundle, probs, {"device": tcfg.device, "epochs_run": len(history), "history": history}


def _plot_confusion(cm: list[list[int]], title: str, out: Path) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_inputs(cfg: DictConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the manifest and return it with the feature matrix this experiment needs."""
    manifest = pd.read_csv(resolve(cfg.data.manifest))
    if cfg.data.split_strategy == "speaker":
        assert_speaker_independent(manifest)
    cache_dir = resolve(cfg.get("cache_dir") or PROCESSED_DIR / "features")
    stats, seq = extract_manifest_features(manifest, cache_dir, n_jobs=cfg.n_jobs)
    return manifest, (stats if cfg.experiment.features == "stats" else seq)


def train_and_evaluate(
    cfg: DictConfig, manifest: pd.DataFrame, X: np.ndarray
) -> tuple[dict[str, Any], Any, dict[str, np.ndarray], dict[str, Any]]:
    """Train on manifest's train split, select on val, report on test."""
    splits = _split_arrays(manifest, X)
    t0 = time.time()
    runner = {"mlp": _run_mlp, "cnn1d": _run_cnn}[cfg.experiment.model]
    model, probs, train_info = runner(cfg, splits)
    train_seconds = time.time() - t0

    test_mask = (manifest["split"] == "test").to_numpy()
    y_test = splits["test"][1]
    report = {
        "experiment": cfg.experiment.name,
        "data": cfg.data.name,
        "split_strategy": cfg.data.split_strategy,
        "seed": cfg.seed,
        "sizes": {s: int(len(splits[s][1])) for s in splits},
        "val": classification_report(splits["val"][1], probs["val"], LABELS),
        "test": classification_report(y_test, probs["test"], LABELS),
        "test_by_gender": group_report(
            y_test, probs["test"], manifest.loc[test_mask, "gender"].reset_index(drop=True), LABELS
        ),
        "test_by_dataset": group_report(
            y_test, probs["test"], manifest.loc[test_mask, "dataset"].reset_index(drop=True), LABELS
        ),
        "timing_seconds": {"train": round(train_seconds, 1)},
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
    }
    return report, model, probs, train_info


def run(cfg: DictConfig) -> dict[str, Any]:
    manifest, X = load_inputs(cfg)
    report, model, probs, train_info = train_and_evaluate(cfg, manifest, X)
    test_mask = (manifest["split"] == "test").to_numpy()
    y_test = manifest.loc[test_mask, "emotion"].map(EMOTION_TO_ID).to_numpy()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_root = resolve(cfg.get("output_root") or ARTIFACTS_DIR / "experiments")
    out_dir = exp_root / f"{cfg.experiment.name}__{cfg.data.name}" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    if "history" in train_info:
        pd.DataFrame(train_info["history"]).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(
        {
            "path": manifest.loc[test_mask, "path"].to_numpy(),
            "true": [LABELS[i] for i in y_test],
            "pred": [LABELS[i] for i in probs["test"].argmax(axis=1)],
            "confidence": probs["test"].max(axis=1).round(4),
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)
    if cfg.experiment.model == "mlp":
        joblib.dump(model, out_dir / "model.joblib")
    else:
        torch.save(model, out_dir / "model.pt")
    _plot_confusion(
        report["test"]["confusion_matrix"],
        f"{cfg.experiment.name} | {cfg.data.name} | test",
        out_dir / "confusion_test.png",
    )

    t = report["test"]
    row = {
        "timestamp": stamp,
        "experiment": cfg.experiment.name,
        "data": cfg.data.name,
        "split": cfg.data.split_strategy,
        "test_wa": round(t["weighted_accuracy"], 4),
        "test_ua": round(t["unweighted_accuracy"], 4),
        "test_macro_f1": round(t["macro_f1"], 4),
        "test_ece": round(t["ece"], 4),
        "run_dir": out_dir.relative_to(exp_root).as_posix(),
    }
    board = exp_root / "leaderboard.csv"
    pd.DataFrame([row]).to_csv(board, mode="a", header=not board.exists(), index=False)

    print(
        json.dumps(
            {
                k: row[k]
                for k in (
                    "experiment",
                    "data",
                    "split",
                    "test_wa",
                    "test_ua",
                    "test_macro_f1",
                    "test_ece",
                )
            },
            indent=2,
        )
    )
    print(f"Run directory: {out_dir}")
    return report


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
