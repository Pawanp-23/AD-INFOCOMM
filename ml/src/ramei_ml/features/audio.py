"""Audio loading and preprocessing shared by every audio model (TRD §3.4)."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ramei_ml.paths import resolve


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    trim_top_db: float = 30.0  # silence removal threshold (the thesis's manual trimming, automated)
    target_rms: float = 0.05  # loudness normalization target
    max_seconds: float = 8.0


def load_audio(path: str | Path, cfg: AudioConfig = AudioConfig()) -> np.ndarray:
    """Load mono audio at cfg.sample_rate, trim leading/trailing silence, normalize loudness."""
    y, _ = librosa.load(resolve(path), sr=cfg.sample_rate, mono=True)
    return preprocess(y, cfg)


def preprocess(y: np.ndarray, cfg: AudioConfig = AudioConfig()) -> np.ndarray:
    y = y.astype(np.float32)
    if y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=cfg.trim_top_db)
    if trimmed.size >= cfg.sample_rate // 10:  # keep original if trimming leaves < 100 ms
        y = trimmed
    y = y[: int(cfg.max_seconds * cfg.sample_rate)]
    rms = float(np.sqrt(np.mean(y**2)))
    if rms > 1e-6:
        y = y * (cfg.target_rms / rms)
    return np.clip(y, -1.0, 1.0)


def fix_length(y: np.ndarray, n_samples: int) -> np.ndarray:
    """Center-crop or zero-pad (right) to exactly n_samples."""
    if y.size >= n_samples:
        start = (y.size - n_samples) // 2
        return y[start : start + n_samples]
    return np.pad(y, (0, n_samples - y.size))
