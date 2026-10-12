"""Signal-quality features per modality for the Reliability Estimation Module (TRD §4.4).

All functions return fixed-length float vectors so they can be cached and normalized.
"""

import math

import numpy as np

AUDIO_QUALITY = ["snr_db", "speech_ratio", "clip_ratio", "log_rms", "duration_s"]
VISUAL_QUALITY = [
    "face_present",
    "det_conf",
    "face_area",
    "log_blur",
    "brightness",
    "brightness_std",
]
TEXT_QUALITY = ["avg_logprob", "no_speech_prob", "compression_ratio", "n_words", "empty"]


def audio_quality(y: np.ndarray, sr: int = 16_000, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Blind SNR estimate from the frame-energy distribution plus simple level statistics.

    SNR ~ loud-frame energy (90th pct) vs quiet-frame energy (10th pct): with stationary noise
    the quiet frames approximate the noise floor.
    """
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    n = 1 + (y.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy = np.mean(y[idx] ** 2, axis=1) + 1e-10
    lo, hi = np.percentile(energy, 10), np.percentile(energy, 90)
    snr = 10 * math.log10(hi / lo)
    speech_ratio = float(np.mean(energy > lo * 10 ** (6 / 10)))  # frames 6 dB above floor
    clip_ratio = float(np.mean(np.abs(y) > 0.99 * np.abs(y).max())) if np.abs(y).max() > 0 else 0.0
    return np.array(
        [snr, speech_ratio, clip_ratio, math.log(np.sqrt(np.mean(y**2)) + 1e-8), y.size / sr],
        dtype=np.float32,
    )


def visual_quality(
    face_present: np.ndarray,
    det_conf: np.ndarray,
    face_area: np.ndarray,
    blur: np.ndarray,
    brightness: np.ndarray,
) -> np.ndarray:
    """Aggregate per-frame face signals (arrays of length n_frames) into one vector."""
    present = face_present > 0

    def mean_present(x: np.ndarray) -> float:
        return float(x[present].mean()) if present.any() else 0.0

    return np.array(
        [
            float(present.mean()),
            mean_present(det_conf),
            mean_present(face_area),
            math.log1p(mean_present(blur)),
            mean_present(brightness),
            float(brightness[present].std()) if present.sum() > 1 else 0.0,
        ],
        dtype=np.float32,
    )


def crops_blur_brightness(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recompute blur and brightness on (possibly corrupted) crops [N, H, W, 3] RGB."""
    import cv2

    from ramei_ml.features.video import blur_score

    grays = [cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in crops]
    return (np.array([blur_score(g) for g in grays]), np.array([g.mean() / 255.0 for g in grays]))


def text_quality(
    avg_logprob: float, no_speech_prob: float, compression_ratio: float, n_words: int
) -> np.ndarray:
    empty = n_words == 0
    return np.array(
        [
            -5.0 if empty or not math.isfinite(avg_logprob) else avg_logprob,
            no_speech_prob,
            compression_ratio,
            math.log1p(n_words),
            float(empty),
        ],
        dtype=np.float32,
    )
