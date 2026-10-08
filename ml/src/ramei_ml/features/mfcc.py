"""Hand-crafted acoustic features for the M0 baselines (TRD §5, stage S0).

- ``utterance_features``: fixed-size statistics vector (thesis replica: MFCC+pitch+energy -> MLP)
- ``mfcc_sequence``:      fixed-length MFCC+Δ+ΔΔ time series (repo replica -> 1D-CNN)
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ramei_ml.features.audio import AudioConfig, fix_length, load_audio


@dataclass(frozen=True)
class FeatureConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    n_mfcc: int = 40
    n_fft: int = 512  # 32 ms at 16 kHz
    hop_length: int = 160  # 10 ms
    n_mels: int = 64
    seq_seconds: float = 3.0  # repo used 3 s clips
    fmin_pitch: float = 65.0
    fmax_pitch: float = 400.0


def _mfcc_stack(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.audio.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    if width < 3:  # too short for deltas
        zeros = np.zeros_like(mfcc)
        return np.concatenate([mfcc, zeros, zeros])
    d1 = librosa.feature.delta(mfcc, width=width, order=1)
    d2 = librosa.feature.delta(mfcc, width=width, order=2)
    return np.concatenate([mfcc, d1, d2])  # (3*n_mfcc, T)


def _mean_std(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])


def utterance_features(y: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Statistics vector: MFCC/Δ/ΔΔ mean+std, pitch, short-term energy, ZCR, spectral centroid."""
    sr = cfg.audio.sample_rate
    if y.size < cfg.n_fft:
        y = np.pad(y, (0, cfg.n_fft - y.size))
    mfcc = _mfcc_stack(y, cfg)
    f0 = librosa.yin(
        y,
        fmin=cfg.fmin_pitch,
        fmax=cfg.fmax_pitch,
        sr=sr,
        frame_length=1024,
        hop_length=cfg.hop_length,
    )
    rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    voiced = f0[(f0 > cfg.fmin_pitch) & (f0 < cfg.fmax_pitch)]
    pitch = np.array([voiced.mean(), voiced.std()]) if voiced.size else np.zeros(2)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[
        0
    ]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )[0] / (sr / 2)
    feats = np.concatenate(
        [
            _mean_std(mfcc),
            np.log1p(pitch),
            [np.log(rms.mean() + 1e-8), np.log(rms.std() + 1e-8)],
            [zcr.mean(), zcr.std(), centroid.mean(), centroid.std()],
        ]
    )
    return feats.astype(np.float32)


def mfcc_sequence(y: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """(3*n_mfcc, T) time series of a fixed-length (cfg.seq_seconds) window."""
    y = fix_length(y, int(cfg.seq_seconds * cfg.audio.sample_rate))
    return _mfcc_stack(y, cfg).astype(np.float32)


def utterance_feature_dim(cfg: FeatureConfig = FeatureConfig()) -> int:
    return 2 * 3 * cfg.n_mfcc + 2 + 2 + 4


def _extract_one(path: str, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    y = load_audio(path, cfg.audio)
    return utterance_features(y, cfg), mfcc_sequence(y, cfg)


def _cache_key(paths: list[str], cfg: FeatureConfig) -> str:
    payload = json.dumps({"paths": paths, "cfg": asdict(cfg)}, sort_keys=True).encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


def extract_manifest_features(
    manifest: pd.DataFrame, cache_dir: Path, cfg: FeatureConfig = FeatureConfig(), n_jobs: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (stats [N, D], sequences [N, C, T]) aligned with manifest rows, cached on disk."""
    paths = manifest["path"].tolist()
    cache = cache_dir / f"mfcc_{_cache_key(paths, cfg)}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["stats"], data["seq"]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_one)(p, cfg) for p in paths
    )
    stats = np.stack([r[0] for r in results])
    seq = np.stack([r[1] for r in results])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, stats=stats, seq=seq)
    return stats, seq
