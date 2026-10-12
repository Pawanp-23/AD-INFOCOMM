"""Download RAVDESS speech video (24 x ~553 MB zips, Zenodo record 1188976) and reduce each
clip to 16 face crops + quality metadata. Videos and zips are deleted after processing, so only
~350 MB stays on disk. Resumable: actors whose 60 clips are processed are skipped.

License: CC BY-NC-SA 4.0 (see download_ravdess.py for the citation).

Usage (from ml/):
    python scripts/download_ravdess_video.py           # -> data/processed/faces/<clip_stem>/
"""

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "src"))

from ramei_ml.features.video import (  # noqa: E402
    FaceConfig,
    FaceDetector,
    extract_faces,
    save_faces,
)

URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip?download=1"
OUT = ML_ROOT / "data" / "processed" / "faces" / "ravdess"
TMP = ML_ROOT / "data" / "raw" / "ravdess_video_tmp"
CLIPS_PER_ACTOR = 60  # speech, full audio-visual (modality 01)


def done(actor: int) -> bool:
    return len(list(OUT.glob(f"01-01-*-{actor:02d}/quality.json"))) >= CLIPS_PER_ACTOR


def download(actor: int) -> Path:
    path = TMP / f"Video_Speech_Actor_{actor:02d}.zip"
    if not path.exists():
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(actor=actor), part)  # noqa: S310 - fixed https URL
        part.replace(path)
    return path


def process(actor: int, archive: Path, detector: FaceDetector, cfg: FaceConfig) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory(dir=TMP) as tmp:
        for member in zf.namelist():
            name = Path(member).name
            if not (name.startswith("01-01-") and name.endswith(".mp4")):
                continue  # keep only full audio-visual speech clips
            out_dir = OUT / Path(name).stem
            if (out_dir / "quality.json").exists():
                count += 1
                continue
            video = Path(zf.extract(member, tmp))
            crops, quality = extract_faces(video, detector, cfg)
            save_faces(out_dir, crops, quality, cfg)
            video.unlink()
            count += 1
    archive.unlink()
    return count


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = FaceConfig()
    detector = FaceDetector(min_confidence=cfg.min_confidence)
    todo = [a for a in range(1, 25) if not done(a)]
    print(f"{24 - len(todo)} actors already processed, {len(todo)} to go", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:  # download ahead while processing
        pending = {a: pool.submit(download, a) for a in todo[:2]}
        for i, actor in enumerate(todo):
            archive = pending.pop(actor).result()
            if i + 2 < len(todo):
                nxt = todo[i + 2]
                pending[nxt] = pool.submit(download, nxt)
            n = process(actor, archive, detector, cfg)
            print(f"actor {actor:02d}: {n} clips processed", flush=True)
    detector.close()
    shutil.rmtree(TMP, ignore_errors=True)
    total = len(list(OUT.glob("*/quality.json")))
    print(f"Done: {total} clips with face crops in {OUT}")


if __name__ == "__main__":
    main()
