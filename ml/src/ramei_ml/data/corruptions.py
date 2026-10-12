"""Input corruptions used to teach and test reliability-aware fusion (TRD §3.5).

Each corruption degrades one modality the way real deployments do: background noise and
clipping for audio, blur / low light / occlusion for faces, ASR-style word errors for text.
"""

import numpy as np


# ----------------------------------------------------------------------------- audio
def colored_noise(n: int, rng: np.random.Generator, color: str = "pink") -> np.ndarray:
    white = rng.normal(size=n)
    if color == "white":
        return white
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0
    spectrum /= np.sqrt(freqs)  # 1/f power
    noise = np.fft.irfft(spectrum, n)
    return noise / (noise.std() + 1e-12)


def add_noise(
    y: np.ndarray, snr_db: float, rng: np.random.Generator, color: str = "pink"
) -> np.ndarray:
    """Additive noise at a target signal-to-noise ratio (dB)."""
    signal_power = np.mean(y**2) + 1e-12
    noise = colored_noise(y.size, rng, color)
    noise *= np.sqrt(signal_power / 10 ** (snr_db / 10)) / (noise.std() + 1e-12)
    return np.clip(y + noise, -1.0, 1.0).astype(np.float32)


def clip_audio(y: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Hard clipping (e.g. an overdriven microphone)."""
    return (np.clip(y, -threshold, threshold) / threshold * 0.5).astype(np.float32)


# ----------------------------------------------------------------------------- video (face crops)
def blur_faces(crops: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    import cv2

    k = int(6 * sigma) | 1
    return np.stack([cv2.GaussianBlur(c, (k, k), sigma) for c in crops])


def darken_faces(
    crops: np.ndarray, factor: float = 0.15, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Low light: scale brightness down and add sensor noise."""
    rng = rng or np.random.default_rng(0)
    dark = crops.astype(np.float32) * factor + rng.normal(0, 4, crops.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def occlude_faces(crops: np.ndarray, rng: np.random.Generator, frac: float = 0.5) -> np.ndarray:
    """Cover a random half of the face (hand, mask, object) in every frame."""
    out = crops.copy()
    n, h, w, _ = out.shape
    lower = rng.random() < 0.5  # lower half (mouth) or upper half (eyes) occluded
    rows = slice(int(h * (1 - frac)), h) if lower else slice(0, int(h * frac))
    out[:, rows, :, :] = rng.integers(0, 60)
    return out


# ----------------------------------------------------------------------------- text
def drop_words(text: str, rate: float, rng: np.random.Generator) -> str:
    """Simulate ASR deletions: drop each word with probability ``rate``."""
    words = text.split()
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept)
