"""Download CREMA-D videos (7,442 .flv clips, ~2.1 GB) and reduce each to 16 face crops.

Clip names come from the already-downloaded AudioWAV folder (run download_crema_d.py first).
Each worker process downloads a clip, extracts faces + quality metadata, and deletes the video,
so ~1.4 GB of JPEG crops stays on disk. Resumable: processed clips are skipped.

License: ODbL 1.0 (see download_crema_d.py for the citation).

Usage (from ml/):
    python scripts/download_crema_d_video.py      # -> data/processed/faces/crema_d/<clip>/
"""

import os
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "src"))

from ramei_ml.features.video import (  # noqa: E402
    FaceConfig,
    FaceDetector,
    extract_faces,
    save_faces,
)

AUDIO_DIR = ML_ROOT / "data" / "raw" / "crema_d" / "AudioWAV"
OUT = ML_ROOT / "data" / "processed" / "faces" / "crema_d"
URLS = [
    "https://gitlab.com/cs-cooper-lab/crema-d-mirror/-/raw/main/VideoFlash/{stem}.flv",
    "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/VideoFlash/{stem}.flv",
]

_detector: FaceDetector | None = None


def _init() -> None:
    global _detector
    os.environ.setdefault("GLOG_minloglevel", "2")
    _detector = FaceDetector(min_confidence=FaceConfig().min_confidence)


def work(stem: str) -> str:
    out_dir = OUT / stem
    if (out_dir / "quality.json").exists():
        return "skip"
    for template in URLS:
        try:
            with urllib.request.urlopen(template.format(stem=stem), timeout=60) as r:  # noqa: S310
                data = r.read()
        except OSError:
            continue
        if not data.startswith(b"FLV"):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / f"{stem}.flv"
            video.write_bytes(data)
            crops, quality = extract_faces(video, _detector, FaceConfig())
        save_faces(out_dir, crops, quality, FaceConfig())
        return "ok"
    return "failed"


def main() -> None:
    stems = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    if not stems:
        raise SystemExit("Run scripts/download_crema_d.py first (clip list comes from AudioWAV).")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (OUT / s / "quality.json").exists()]
    print(f"{len(stems)} clips, {len(stems) - len(todo)} done, {len(todo)} to go", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init) as pool:
        futures = {pool.submit(work, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            if fut.result() == "failed":
                failed.append(futures[fut])
            if i % 250 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} processed, {len(failed)} failed", flush=True)
    total = len(list(OUT.glob("*/quality.json")))
    print(f"Done: {total}/{len(stems)} clips with face crops in {OUT}")
    if failed:
        print(f"Failed ({len(failed)}), re-run to retry: {failed[:10]}")


if __name__ == "__main__":
    main()
