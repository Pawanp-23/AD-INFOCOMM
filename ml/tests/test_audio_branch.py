"""Audio branch tests with a tiny randomly initialized WavLM (no download needed)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ramei_ml.data.audio_dataset import (
    AugmentConfig,
    WaveformDataset,
    build_waveform_cache,
    collate_waveforms,
)
from ramei_ml.data.manifest import build_manifest
from ramei_ml.evaluation.calibration import fit_temperature, softmax
from ramei_ml.evaluation.metrics import expected_calibration_error
from ramei_ml.models.audio_branch import AudioBranch, AudioBranchConfig
from ramei_ml.models.pooling import AttentiveStatsPooling, lengths_to_mask
from ramei_ml.training.audio_branch import run

TINY_ENCODER = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": [32] * 7,
    "num_conv_pos_embeddings": 16,
    "num_conv_pos_embedding_groups": 2,
}


def tiny_cfg(mode: str = "lora") -> AudioBranchConfig:
    return AudioBranchConfig(
        pretrained=None, encoder_config=TINY_ENCODER, mode=mode, embed_dim=16, head_hidden=16
    )


def test_attentive_pooling_ignores_padding() -> None:
    torch.manual_seed(0)
    pool = AttentiveStatsPooling(dim=8).eval()
    x = torch.randn(1, 5, 8)
    padded = torch.cat([x, torch.randn(1, 3, 8) * 100], dim=1)  # garbage in the padding
    a = pool(x, lengths_to_mask(torch.tensor([5]), 5))
    b = pool(padded, lengths_to_mask(torch.tensor([5]), 8))
    torch.testing.assert_close(a, b)
    assert a.shape == (1, 16)


@pytest.mark.parametrize("mode", ["lora", "frozen"])
def test_audio_branch_shapes_and_trainable_params(mode: str) -> None:
    model = AudioBranch(tiny_cfg(mode))
    waves = torch.randn(2, 16_000) * 0.1
    out = model(waves, torch.tensor([16_000, 9_000]))
    assert out["z"].shape == (2, 16)
    assert out["logits"].shape == (2, 7)
    assert torch.isclose(out["layer_weights"].sum(), torch.tensor(1.0))

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    encoder_trainable = {n for n in trainable if n.startswith("encoder.")}
    if mode == "frozen":
        assert not encoder_trainable
    else:
        assert encoder_trainable and all("lora_" in n for n in encoder_trainable)
    assert set(model.trainable_state_dict()) == trainable


def test_collate_pads_and_reports_lengths() -> None:
    waves, lengths, labels = collate_waveforms([(torch.ones(3), 1), (torch.ones(5), 2)])
    assert waves.shape == (2, 5)
    assert lengths.tolist() == [3, 5]
    assert waves[0, 3:].abs().sum() == 0
    assert labels.tolist() == [1, 2]


def test_waveform_cache_and_dataset(synthetic_ravdess: Path, tmp_path: Path) -> None:
    df = build_manifest({"ravdess": synthetic_ravdess}).head(4)
    files = build_waveform_cache(df["path"].tolist(), tmp_path, n_jobs=1)
    assert all(f.exists() for f in files)
    ds = WaveformDataset(df, files, train=True, augment=AugmentConfig(max_seconds=0.5))
    wave, label = ds[0]
    assert wave.dtype == torch.float32 and wave.numel() <= 8_000
    assert 0 <= label < 7


def test_temperature_scaling_fixes_overconfidence() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.0  # ~55% accurate ...
    logits *= 6.0  # ... but extremely confident
    t = fit_temperature(logits, y)
    assert t > 2.0
    assert (
        expected_calibration_error(softmax(logits, t), y)
        < expected_calibration_error(softmax(logits), y) / 2
    )


def test_audio_branch_run_end_to_end(synthetic_ravdess: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest({"ravdess": synthetic_ravdess}).to_csv(manifest_path, index=False)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "n_jobs": 1,
            "cv_folds": 3,
            "max_folds": 1,
            "cache_dir": str(tmp_path / "cache"),
            "output_root": str(tmp_path / "out"),
            "data": {
                "name": "synthetic",
                "manifest": str(manifest_path),
                "split_strategy": "speaker",
            },
            "experiment": {
                "name": "e2_test",
                "model": {
                    "pretrained": None,
                    "encoder_config": TINY_ENCODER,
                    "mode": "lora",
                    "embed_dim": 16,
                    "head_hidden": 16,
                },
                "train": {
                    "epochs": 1,
                    "batch_size": 8,
                    "grad_accum": 1,
                    "eval_batch_size": 8,
                    "lr_encoder": 1e-3,
                    "lr_head": 1e-3,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "label_smoothing": 0.1,
                    "max_grad_norm": 1.0,
                    "patience": 1,
                    "num_workers": 0,
                    "augment": {
                        "max_seconds": 1.0,
                        "gain_db": 3.0,
                        "noise_prob": 0.0,
                        "noise_snr_db": [15.0, 35.0],
                    },
                },
            },
        }
    )
    result = run(cfg)
    assert result["folds_run"] == 1
    assert 0.0 <= result["summary"]["macro_f1"]["mean"] <= 1.0
    fold_dir = next((tmp_path / "out").glob("*/*/fold0"))
    assert {"trainable_state.pt", "test_logits.npz", "metrics.json"} <= {
        p.name for p in fold_dir.iterdir()
    }


def test_training_mode_always_returns_all_layers() -> None:
    """LayerDrop would randomly drop hidden states and break the layer-weighted sum."""
    torch.manual_seed(0)
    model = AudioBranch(tiny_cfg("lora")).train()
    assert model.encoder.config.layerdrop == 0.0
    for _ in range(20):
        model(torch.randn(2, 8_000) * 0.1, torch.tensor([8_000, 8_000]))
