"""Speaker-independent k-fold cross-validation for the S0 baselines.

A single 4-speaker test set is noisy, so baselines are reported as mean +/- std over K folds.
Speakers are dealt into K gender-balanced folds; fold k is test, fold k+1 is validation,
the rest is train. Every speaker is tested exactly once.

Run from ml/:
    python -m ramei_ml.training.cv                         # E0, 6 folds
    python -m ramei_ml.training.cv experiment=e1_cnn cv_folds=6
"""

import json
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ramei_ml.paths import ARTIFACTS_DIR, resolve
from ramei_ml.training.baseline import load_inputs, train_and_evaluate

METRICS = ("weighted_accuracy", "unweighted_accuracy", "macro_f1", "ece")


def speaker_folds(manifest: pd.DataFrame, k: int, seed: int = 42) -> dict[str, int]:
    """Map speaker_id -> fold, dealing each (dataset, gender) stratum round-robin."""
    rng = np.random.default_rng(seed)
    speakers = manifest[["dataset", "speaker_id", "gender"]].drop_duplicates()
    speakers = speakers.assign(gender=speakers["gender"].fillna("unknown"))
    fold_of: dict[str, int] = {}
    offset = 0
    for _, group in speakers.groupby(["dataset", "gender"], sort=True):
        ids = sorted(group["speaker_id"])
        rng.shuffle(ids)
        for i, spk in enumerate(ids):
            fold_of[spk] = (offset + i) % k
        offset += len(ids)  # continue dealing where the previous stratum stopped
    return fold_of


def fold_manifest(
    manifest: pd.DataFrame, fold_of: dict[str, int], fold: int, k: int
) -> pd.DataFrame:
    folds = manifest["speaker_id"].map(fold_of)
    split = np.where(folds == fold, "test", np.where(folds == (fold + 1) % k, "val", "train"))
    return manifest.assign(split=split)


def run_cv(cfg: DictConfig) -> dict[str, Any]:
    if cfg.data.split_strategy != "speaker":
        raise ValueError("Cross-validation is only meaningful with speaker-independent data")
    k = int(cfg.cv_folds)
    manifest, X = load_inputs(cfg)
    fold_of = speaker_folds(manifest, k, cfg.seed)

    folds = []
    for fold in range(k):
        fm = fold_manifest(manifest, fold_of, fold, k)
        report, *_ = train_and_evaluate(cfg, fm, X)
        test_speakers = sorted(fm.loc[fm["split"] == "test", "speaker_id"].unique())
        folds.append(
            {
                "fold": fold,
                "test_speakers": test_speakers,
                **{m: report["test"][m] for m in METRICS},
                "by_gender": report["test_by_gender"],
            }
        )
        print(
            f"fold {fold}: macro_f1={report['test']['macro_f1']:.4f} "
            f"wa={report['test']['weighted_accuracy']:.4f} speakers={test_speakers}"
        )

    summary = {
        m: {
            "mean": float(np.mean([f[m] for f in folds])),
            "std": float(np.std([f[m] for f in folds])),
        }
        for m in METRICS
    }
    gender_f1: dict[str, list[float]] = {}
    for f in folds:
        for g, v in f["by_gender"].items():
            gender_f1.setdefault(g, []).append(v["macro_f1"])
    result = {
        "experiment": cfg.experiment.name,
        "data": cfg.data.name,
        "folds_k": k,
        "seed": cfg.seed,
        "summary": summary,
        "macro_f1_by_gender": {g: float(np.mean(v)) for g, v in gender_f1.items()},
        "folds": folds,
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_root = resolve(cfg.get("output_root") or ARTIFACTS_DIR / "experiments")
    out_dir = exp_root / f"{cfg.experiment.name}__{cfg.data.name}_cv{k}" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    row = {
        "timestamp": stamp,
        "experiment": cfg.experiment.name,
        "data": f"{cfg.data.name}_cv{k}",
        "split": f"speaker_cv{k}",
        "test_wa": round(summary["weighted_accuracy"]["mean"], 4),
        "test_ua": round(summary["unweighted_accuracy"]["mean"], 4),
        "test_macro_f1": round(summary["macro_f1"]["mean"], 4),
        "test_ece": round(summary["ece"]["mean"], 4),
        "run_dir": out_dir.relative_to(exp_root).as_posix(),
    }
    board = exp_root / "leaderboard.csv"
    pd.DataFrame([row]).to_csv(board, mode="a", header=not board.exists(), index=False)

    print("\n" + "  ".join(f"{m}={v['mean']:.4f}+/-{v['std']:.4f}" for m, v in summary.items()))
    print(f"Run directory: {out_dir}")
    return result


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    run_cv(cfg)


if __name__ == "__main__":
    main()
