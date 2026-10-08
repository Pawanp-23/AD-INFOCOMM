"""End-to-end smoke tests of stage S0 on the synthetic dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from ramei_ml.data.manifest import build_manifest
from ramei_ml.models.baseline import CNN1D
from ramei_ml.training.baseline import run

EXPERIMENTS = {
    "mlp": {
        "name": "e0_test",
        "model": "mlp",
        "features": "stats",
        "mlp": {
            "hidden_layer_sizes": [32],
            "alpha": 0.01,
            "batch_size": 32,
            "learning_rate": "adaptive",
            "max_iter": 300,
        },
    },
    "cnn1d": {
        "name": "e1_test",
        "model": "cnn1d",
        "features": "seq",
        "cnn": {
            "epochs": 3,
            "batch_size": 16,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "patience": 3,
            "dropout": 0.1,
        },
    },
}


def test_cnn_forward_shape() -> None:
    model = CNN1D(in_channels=120, num_classes=7)
    assert model(torch.randn(2, 120, 301)).shape == (2, 7)


@pytest.mark.parametrize("model", ["mlp", "cnn1d"])
def test_baseline_run_end_to_end(model: str, synthetic_ravdess: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest({"ravdess": synthetic_ravdess}, seed=3).to_csv(manifest_path, index=False)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "n_jobs": 1,
            "cache_dir": str(tmp_path / "cache"),
            "output_root": str(tmp_path / "out"),
            "data": {
                "name": "synthetic",
                "manifest": str(manifest_path),
                "split_strategy": "speaker",
            },
            "experiment": EXPERIMENTS[model],
        }
    )

    report = run(cfg)

    assert 0.0 <= report["test"]["macro_f1"] <= 1.0
    assert set(report["test_by_gender"]) == {"male", "female"}
    if model == "mlp":  # the synthetic tones are easily separable
        assert report["test"]["weighted_accuracy"] > 0.6
    run_dirs = list((tmp_path / "out").glob("*/*"))
    assert len(run_dirs) == 1
    files = {p.name for p in run_dirs[0].iterdir()}
    assert {"metrics.json", "config.yaml", "test_predictions.csv", "confusion_test.png"} <= files
    assert json.loads((run_dirs[0] / "metrics.json").read_text())["split_strategy"] == "speaker"
    board = pd.read_csv(tmp_path / "out" / "leaderboard.csv")
    assert len(board) == 1
