"""Canonical filesystem locations inside ml/ (independent of the current working directory).

RAMEI_ML_ROOT overrides the root, e.g. in containers where the package is installed into
site-packages and data (downloaded models) must live in a writable volume.
"""

import os
from pathlib import Path

ML_ROOT = Path(os.environ.get("RAMEI_ML_ROOT") or Path(__file__).resolve().parents[2])
CONFIG_DIR = ML_ROOT / "configs"
DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
ARTIFACTS_DIR = ML_ROOT / "artifacts"


def resolve(path: str | Path) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are taken from ml/."""
    p = Path(path)
    return p if p.is_absolute() else ML_ROOT / p
