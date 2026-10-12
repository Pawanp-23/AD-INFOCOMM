"""Run the whole M2 pipeline in order, logging each step (resumable: finished steps are skipped).

python scripts/run_m2.py            # extract features -> cross-validate fusion -> export bundle
"""

import subprocess
import sys
import time
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
STORE = ML_ROOT / "data" / "processed" / "fusion" / "crema_d"
BUNDLES = ML_ROOT / "artifacts" / "bundles"


def step(name: str, args: list[str], done: Path | None = None) -> None:
    if done is not None and done.exists():
        print(f"[skip] {name}", flush=True)
        return
    print(f"[run ] {name}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *args], cwd=ML_ROOT, check=True)
    print(f"[done] {name} in {time.time() - t0:.0f}s", flush=True)


def latest_results() -> Path | None:
    runs = sorted(
        (ML_ROOT / "artifacts" / "experiments" / "m2_fusion_crema_d").glob("*/results.json")
    )
    return runs[-1] if runs else None


def main() -> None:
    step("index", ["-m", "ramei_ml.fusion.extract", "index"], STORE / "clips.csv")
    step("audio features", ["-m", "ramei_ml.fusion.extract", "audio"], STORE / "audio_snr0.npz")
    step("speech recognition", ["-m", "ramei_ml.fusion.extract", "asr"], STORE / "transcripts.csv")
    step("text features", ["-m", "ramei_ml.fusion.extract", "text"], STORE / "text_drop50.npz")
    step(
        "visual features", ["-m", "ramei_ml.fusion.extract", "visual"], STORE / "visual_occlude.npz"
    )
    results = latest_results()
    if results is None:
        step("fusion cross-validation", ["-m", "ramei_ml.fusion.train"])
        results = latest_results()
    step(
        "export bundle",
        ["-m", "ramei_ml.fusion.export", "--version", "1.0.0", "--cv-results", str(results)],
        BUNDLES / "ramei-rgcmt-1.0.0" / "manifest.json",
    )
    print("M2 PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
