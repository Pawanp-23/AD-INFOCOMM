from pathlib import Path

import numpy as np
import pytest

from ramei_ml.data.manifest import build_manifest
from ramei_ml.evaluation.metrics import classification_report, expected_calibration_error
from ramei_ml.features.audio import AudioConfig, fix_length, preprocess
from ramei_ml.features.mfcc import (
    FeatureConfig,
    extract_manifest_features,
    mfcc_sequence,
    utterance_feature_dim,
    utterance_features,
)

LABELS = ["a", "b", "c"]


def test_preprocess_trims_silence_and_normalizes() -> None:
    sr = 16_000
    tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    y = np.concatenate([np.zeros(sr), tone, np.zeros(sr)]).astype(np.float32)
    out = preprocess(y, AudioConfig())
    assert out.size < y.size * 0.6
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0.05, rel=0.05)


def test_fix_length() -> None:
    assert fix_length(np.ones(10), 4).shape == (4,)
    padded = fix_length(np.ones(3), 5)
    assert padded.tolist() == [1, 1, 1, 0, 0]


def test_feature_shapes() -> None:
    cfg = FeatureConfig()
    y = np.random.default_rng(0).normal(0, 0.05, 20_000).astype(np.float32)
    assert utterance_features(y, cfg).shape == (utterance_feature_dim(cfg),)
    seq = mfcc_sequence(y, cfg)
    assert seq.shape[0] == 3 * cfg.n_mfcc
    assert seq.shape[1] == int(cfg.seq_seconds * 16_000) // cfg.hop_length + 1


def test_features_handle_very_short_audio() -> None:
    feats = utterance_features(np.zeros(100, dtype=np.float32))
    assert np.isfinite(feats).all()


def test_extract_manifest_features_caches(synthetic_ravdess: Path, tmp_path: Path) -> None:
    df = build_manifest({"ravdess": synthetic_ravdess}).head(6)
    stats, seq = extract_manifest_features(df, tmp_path, n_jobs=1)
    assert stats.shape[0] == seq.shape[0] == 6
    assert len(list(tmp_path.glob("mfcc_*.npz"))) == 1
    stats2, _ = extract_manifest_features(df, tmp_path, n_jobs=1)  # served from cache
    np.testing.assert_array_equal(stats, stats2)


def test_ece_zero_for_perfect_one_hot() -> None:
    y = np.array([0, 1, 2, 1])
    probs = np.eye(3)[y]
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_ece_overconfident_model() -> None:
    y = np.array([0, 1, 0, 1])
    probs = np.tile([0.9, 0.1, 0.0], (4, 1))  # 90% confident, 50% accurate -> ECE 0.4
    assert expected_calibration_error(probs, y) == pytest.approx(0.4)


def test_classification_report_values() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[[0, 0, 1, 0, 2, 2]] * 0.8 + 0.2 / 3
    rep = classification_report(y, probs, LABELS)
    assert rep["weighted_accuracy"] == pytest.approx(5 / 6)
    assert rep["unweighted_accuracy"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert rep["confusion_matrix"][1] == [1, 1, 0]
    assert set(rep["per_class_f1"]) == set(LABELS)
