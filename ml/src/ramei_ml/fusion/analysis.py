"""Post-hoc analysis of a fusion CV run from its saved fold checkpoints (no retraining).

Per scenario: per-gender macro-F1 (NFR-09), and selective prediction under the abstention
threshold — coverage and accuracy on the segments the model commits to (FR-14). Also the
pooled clean-input confusion matrix.

Run from ml/:
    python -m ramei_ml.fusion.analysis artifacts/experiments/m2_fusion_crema_d/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from ramei_ml.data.labels import EMOTIONS
from ramei_ml.evaluation.calibration import softmax
from ramei_ml.fusion.data import FeatureStore
from ramei_ml.fusion.model import FusionConfig, RAMEIFusion
from ramei_ml.fusion.train import DEVICE, predict
from ramei_ml.training.cv import fold_manifest, speaker_folds

SCENARIOS = ("clean", "noisy_room_0db", "visual_dark", "camera_off", "noisy_room_camera_off")
THRESHOLD = 0.45
NUMPY_GLOBALS = [
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype("float32")),
    type(np.dtype("float64")),
]


def _f1(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return round(float(f1_score(y[mask], pred[mask], average="macro")), 4)


def analyse(run_dir: Path, experiment: str, folds: int, seed: int) -> dict:
    store = FeatureStore()
    fold_of = speaker_folds(store.clips, folds, seed)
    pooled: dict[str, dict[str, list]] = {s: {"y": [], "p": [], "g": []} for s in SCENARIOS}
    for f in range(folds):
        # Safe loader; the checkpoint's only non-tensor payload is numpy quality statistics.
        with torch.serialization.safe_globals(NUMPY_GLOBALS):
            ckpt = torch.load(run_dir / f"{experiment}_fold{f}.pt", map_location=DEVICE)
        model = RAMEIFusion(FusionConfig(use_reliability=ckpt["use_reliability"])).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        store.q_stats = ckpt["q_stats"]  # quality normalization fitted on this fold's train split
        fm = fold_manifest(store.clips, fold_of, f, folds)
        test = np.flatnonzero((fm["split"] == "test").to_numpy())
        for s in SCENARIOS:
            r = predict(model, store, test, store.scenario_variants(len(test), s))
            pooled[s]["y"].append(store.y[test])
            pooled[s]["p"].append(softmax(r["logits"], ckpt["temperature"]))
            pooled[s]["g"].append(store.clips["gender"].to_numpy()[test])

    out: dict = {"experiment": experiment, "threshold": THRESHOLD, "scenarios": {}}
    for s, d in pooled.items():
        y, p, g = (np.concatenate(d[k]) for k in ("y", "p", "g"))
        pred, conf = p.argmax(1), p.max(1)
        keep = conf >= THRESHOLD
        out["scenarios"][s] = {
            "macro_f1_pooled": _f1(y, pred, np.ones_like(y, dtype=bool)),
            "macro_f1_by_gender": {gender: _f1(y, pred, g == gender) for gender in sorted(set(g))},
            "accuracy_all": round(float((pred == y).mean()), 4),
            "coverage": round(float(keep.mean()), 4),
            "selective_accuracy": round(float((pred[keep] == y[keep]).mean()), 4),
            "abstained_accuracy": round(float((pred[~keep] == y[~keep]).mean()), 4)
            if (~keep).any()
            else None,
        }
        if s == "clean":
            out["confusion_clean"] = {
                "labels": list(EMOTIONS),
                "matrix": confusion_matrix(y, pred, labels=range(len(EMOTIONS))).tolist(),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--experiment", default="e8_rgcmt")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = analyse(args.run_dir, args.experiment, args.folds, args.seed)
    path = args.run_dir / f"analysis_{args.experiment}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["scenarios"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
