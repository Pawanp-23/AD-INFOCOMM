"""Shared fixtures: a tiny synthetic dataset in RAVDESS naming format (no real data needed)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 16_000
# RAVDESS emotion codes used in the fixture (calm=02 omitted to keep classes balanced)
CODES = ["01", "03", "04", "05", "06", "07", "08"]


def _tone(code: str, rng: np.random.Generator) -> np.ndarray:
    """Emotion-dependent pitch + loudness so a model can actually learn something."""
    idx = CODES.index(code)
    t = np.arange(int(SR * rng.uniform(1.2, 2.0))) / SR
    f0 = 120 + 35 * idx + rng.normal(0, 4)
    y = (0.2 + 0.05 * idx) * np.sin(2 * np.pi * f0 * t) + 0.01 * rng.normal(size=t.size)
    pad = np.zeros(int(0.2 * SR))  # leading/trailing silence for the trimming step
    return np.concatenate([pad, y, pad]).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_ravdess(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ravdess")
    rng = np.random.default_rng(0)
    for actor in range(1, 13):  # 12 actors: 6 male (odd), 6 female (even)
        actor_dir = root / f"Actor_{actor:02d}"
        actor_dir.mkdir()
        for code in CODES:
            for rep in ("01", "02"):
                name = f"03-01-{code}-01-01-{rep}-{actor:02d}.wav"
                sf.write(actor_dir / name, _tone(code, rng), SR)
        # files that must be ignored: song channel and an unrelated file
        sf.write(actor_dir / f"03-02-03-01-01-01-{actor:02d}.wav", _tone("03", rng), SR)
        (actor_dir / "notes.txt").write_text("ignore me")
    return root
