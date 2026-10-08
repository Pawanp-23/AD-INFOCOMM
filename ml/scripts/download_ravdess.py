"""Download RAVDESS speech audio (Audio_Speech_Actors_01-24.zip, ~208 MB) from Zenodo.

License: CC BY-NC-SA 4.0 — academic / non-commercial use, attribution required:
  Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and
  Song (RAVDESS). PLoS ONE 13(5): e0196391. https://doi.org/10.5281/zenodo.1188976

Usage (from ml/):
    python scripts/download_ravdess.py            # -> data/raw/ravdess/Actor_XX/*.wav
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
DEST = ML_ROOT / "data" / "raw" / "ravdess"
EXPECTED_WAVS = 1440  # 24 actors x 60 speech files


def _progress(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    if total > 0:
        pct = min(100.0, 100 * done / total)
        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()


def main() -> None:
    existing = len(list(DEST.rglob("*.wav"))) if DEST.exists() else 0
    if existing >= EXPECTED_WAVS:
        print(f"RAVDESS already present ({existing} wav files) at {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST.parent / "Audio_Speech_Actors_01-24.zip"
    if not archive.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive, _progress)  # noqa: S310 - fixed https URL
        print()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"Corrupt archive member: {bad}. Delete {archive} and retry.")
        for member in zf.namelist():  # guard against path traversal
            if not (DEST / member).resolve().is_relative_to(DEST.resolve()):
                raise SystemExit(f"Unsafe path in archive: {member}")
        zf.extractall(DEST)
    count = len(list(DEST.rglob("*.wav")))
    print(f"Extracted {count} wav files to {DEST}")
    if count != EXPECTED_WAVS:
        print(f"WARNING: expected {EXPECTED_WAVS} files")
    archive.unlink()


if __name__ == "__main__":
    main()
