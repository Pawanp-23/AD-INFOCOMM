"""Download CREMA-D audio (AudioWAV, 7,442 clips, ~600 MB) + VideoDemographics.csv.

License: Open Database License (ODbL 1.0); contents under the Database Contents License.
  Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014). CREMA-D: Crowd-sourced
  Emotional Multimodal Actors Dataset. IEEE Trans. Affective Computing 5(4): 377-390.

The repository keeps media in Git LFS (a full clone is ~7.5 GB, mostly video), and LFS batch
transfers from the mirrors are unreliable. So this script:
  1. makes a blob-less sparse clone of AudioWAV/ + VideoDemographics.csv (LFS smudge disabled),
     which yields one small LFS pointer file (with the content's sha256) per clip;
  2. downloads each clip over HTTPS in parallel and verifies it against the pointer's sha256.
Re-running resumes: verified files are skipped. Requires git.

Usage (from ml/):
    python scripts/download_crema_d.py          # -> data/raw/crema_d/AudioWAV/*.wav
"""

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
DEST = ML_ROOT / "data" / "raw" / "crema_d"
CLONE_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"  # authors' recommended mirror
RAW_URLS = [  # tried in order for each file
    "https://gitlab.com/cs-cooper-lab/crema-d-mirror/-/raw/main/AudioWAV/{name}",
    "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{name}",
]
EXPECTED_WAVS = 7442
POINTER_PREFIX = b"version https://git-lfs"


def git(*args: str, cwd: Path | None = None) -> None:
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True)


def sparse_clone() -> None:
    if (DEST / ".git").exists():
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DEST, ignore_errors=True)
    print(f"Sparse clone of {CLONE_URL} (pointers only)")
    git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", CLONE_URL, str(DEST))
    git("sparse-checkout", "set", "--no-cone", "/AudioWAV/", "/VideoDemographics.csv", cwd=DEST)
    git("checkout", cwd=DEST)


def pointer_oid(path: Path) -> str | None:
    """sha256 from an LFS pointer file, or None if the file is already real content."""
    head = path.read_bytes()[:200]
    if not head.startswith(POINTER_PREFIX):
        return None
    for line in head.decode().splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Malformed LFS pointer: {path}")


def fetch(path: Path, oid: str) -> str:
    for template in RAW_URLS:
        url = template.format(name=path.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https hosts
                data = resp.read()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() == oid:
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return "ok"
    return "failed"


def main() -> None:
    sparse_clone()
    wavs = sorted((DEST / "AudioWAV").glob("*.wav"))
    todo = [(p, oid) for p in wavs if (oid := pointer_oid(p)) is not None]
    done = len(wavs) - len(todo)
    print(f"{len(wavs)} clips listed, {done} already downloaded, {len(todo)} to go")

    failed = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch, p, oid): p for p, oid in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            if fut.result() != "ok":
                failed += 1
                print(f"\nFAILED: {futures[fut].name}")
            if i % 200 == 0 or i == len(todo):
                sys.stdout.write(f"\r  {i}/{len(todo)} downloaded")
                sys.stdout.flush()
    print()

    remaining = sum(pointer_oid(p) is not None for p in wavs)
    print(f"{len(wavs)} wav files, {remaining} not yet downloaded")
    if len(wavs) != EXPECTED_WAVS or remaining or failed:
        raise SystemExit("Download incomplete — re-run the script to resume.")


if __name__ == "__main__":
    main()
