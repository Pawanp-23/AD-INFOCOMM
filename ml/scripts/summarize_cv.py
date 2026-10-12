"""Pool cross-validation results across runs (e.g. seeds) of the same experiment.

Every fold of every run counts as one sample, so with 3 seeds x 6 folds the mean and std are
over 18 test sets. Also reports the seed-to-seed spread of the per-run means.

Usage (from ml/):
    python scripts/summarize_cv.py artifacts/experiments/e2_audio_wavlm_lora__ravdess_cv6
"""

import json
import sys
from pathlib import Path

import numpy as np

METRICS = ("weighted_accuracy", "unweighted_accuracy", "macro_f1", "ece_raw", "ece")


def load_runs(exp_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        folds = sorted(run_dir.glob("fold*/metrics.json"))
        if not (run_dir / "cv_metrics.json").exists() or not folds:
            continue  # incomplete or crashed run
        cv = json.loads((run_dir / "cv_metrics.json").read_text())
        if cv.get("folds_run", len(folds)) != cv.get("folds_k", len(folds)):
            continue  # partial run (e.g. a max_folds pilot)
        tests = [json.loads(f.read_text())["test"] for f in folds]
        runs.append(
            {
                "dir": run_dir.name,
                "seed": cv["seed"],
                "tests": tests,
                "by_gender": cv.get("macro_f1_by_gender", {}),
            }
        )
    return runs


def main() -> None:
    exp_dir = Path(sys.argv[1])
    runs = load_runs(exp_dir)
    if not runs:
        raise SystemExit(f"No completed runs in {exp_dir}")
    out: dict = {
        "experiment_dir": exp_dir.name,
        "runs": len(runs),
        "seeds": [r["seed"] for r in runs],
        "test_sets": sum(len(r["tests"]) for r in runs),
        "pooled": {},
        "seed_spread": {},
    }
    for m in METRICS:
        values = [t[m] for r in runs for t in r["tests"] if m in t]
        run_means = [np.mean([t[m] for t in r["tests"] if m in t]) for r in runs]
        out["pooled"][m] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
        }
        out["seed_spread"][m] = round(float(np.max(run_means) - np.min(run_means)), 4)
    genders = {g for r in runs for g in r["by_gender"]}
    out["macro_f1_by_gender"] = {
        g: round(float(np.mean([r["by_gender"][g] for r in runs if g in r["by_gender"]])), 4)
        for g in sorted(genders)
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
