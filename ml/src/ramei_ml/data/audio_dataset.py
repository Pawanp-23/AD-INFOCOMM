"""Waveform dataset for the audio branch (TRD §4.1).

Preprocessed 16 kHz waveforms are cached as int16 .npy files, so expensive resampling/trimming
runs once. Batches are right-padded; true lengths are returned for masking.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from ramei_ml.data.labels import EMOTION_TO_ID
from ramei_ml.features.audio import AudioConfig, load_audio


def _cache_path(cache_dir: Path, path: str, cfg: AudioConfig) -> Path:
    key = f"{path}|{cfg.sample_rate}|{cfg.trim_top_db}|{cfg.target_rms}|{cfg.max_seconds}"
    return cache_dir / f"{hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:20]}.npy"


def _materialize(path: str, out: Path, cfg: AudioConfig) -> None:
    if not out.exists():
        y = load_audio(path, cfg)
        np.save(out, (y * 32767).astype(np.int16))


def build_waveform_cache(
    paths: list[str], cache_dir: Path, cfg: AudioConfig = AudioConfig(), n_jobs: int = -1
) -> list[Path]:
    """Preprocess every file once; return the cache path for each input path (same order)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    outs = [_cache_path(cache_dir, p, cfg) for p in paths]
    todo = [(p, o) for p, o in zip(paths, outs, strict=True) if not o.exists()]
    if todo:
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_materialize)(p, o, cfg) for p, o in todo
        )
    return outs


@dataclass
class AugmentConfig:
    max_seconds: float = 6.0  # random crop length for training
    gain_db: float = 6.0  # random gain range ±gain_db
    noise_prob: float = 0.0  # additive white noise probability (corruptions come in M2)
    noise_snr_db: tuple[float, float] = (15.0, 35.0)


class WaveformDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_files: list[Path],
        sample_rate: int = 16_000,
        train: bool = False,
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.files = list(cache_files)
        self.labels = manifest["emotion"].map(EMOTION_TO_ID).to_numpy()
        self.sample_rate = sample_rate
        self.train = train
        self.augment = augment or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        y = np.load(self.files[idx]).astype(np.float32) / 32767.0
        if self.train:
            y = self._augment(y)
        return torch.from_numpy(y), int(self.labels[idx])

    def _augment(self, y: np.ndarray) -> np.ndarray:
        a = self.augment
        max_len = int(a.max_seconds * self.sample_rate)
        if y.size > max_len:
            start = int(self.rng.integers(0, y.size - max_len + 1))
            y = y[start : start + max_len]
        y = y * 10 ** (self.rng.uniform(-a.gain_db, a.gain_db) / 20)
        if a.noise_prob and self.rng.random() < a.noise_prob:
            snr = self.rng.uniform(*a.noise_snr_db)
            power = np.mean(y**2) + 1e-10
            y = y + self.rng.normal(0, np.sqrt(power / 10 ** (snr / 10)), y.size)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


def collate_waveforms(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to the longest clip. Returns (waves [B, T], lengths [B], labels [B])."""
    lengths = torch.tensor([w.numel() for w, _ in batch], dtype=torch.long)
    waves = torch.zeros(len(batch), int(lengths.max()))
    for i, (w, _) in enumerate(batch):
        waves[i, : w.numel()] = w
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return waves, lengths, labels


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker its own reproducible augmentation stream.

    Without this every worker inherits a copy of the same RNG state.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and isinstance(info.dataset, WaveformDataset):
        info.dataset.rng = np.random.default_rng(info.seed % 2**32)
