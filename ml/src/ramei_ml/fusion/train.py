"""Stage S2-S4 — reliability module + adaptive fusion + calibration, speaker-independent CV.

Experiments (TRD §6.2), all trained per fold on the same frozen-encoder features:
  e8_rgcmt          full RAMEI: corruption + modality-dropout training, reliability-gated fusion
  e9_no_reliability ablation: same training, fusion attention without the reliability bias
  e10_clean_train   ablation: trained on clean, complete inputs only
From the e8 model we also read out, per scenario:
  e5_static         equal-weight average of the available branch probabilities
  e6_rel_late       reliability-weighted late fusion  p = sum r_m p_m / sum r_m
  branch_*          each branch alone

Run from ml/:
    python -m ramei_ml.fusion.train                    # all experiments, 5 folds
    python -m ramei_ml.fusion.train --experiments e8_rgcmt --folds 5
"""

import argparse
import json
import math
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from ramei_ml.data.labels import EMOTIONS, NUM_CLASSES
from ramei_ml.evaluation.calibration import fit_temperature, softmax
from ramei_ml.evaluation.metrics import classification_report
from ramei_ml.fusion.data import SCENARIOS, FeatureStore, TrainSampling
from ramei_ml.fusion.model import MODALITIES, FusionConfig, RAMEIFusion, fusion_loss
from ramei_ml.paths import ARTIFACTS_DIR
from ramei_ml.training.cv import fold_manifest, speaker_folds
from ramei_ml.training.torch_trainer import class_weights

LABELS = list(EMOTIONS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS = {
    "e8_rgcmt": {"use_reliability": True, "sampling": TrainSampling()},
    "e9_no_reliability": {"use_reliability": False, "sampling": TrainSampling()},
    "e10_clean_train": {
        "use_reliability": True,
        "sampling": TrainSampling(p_clean=1.0, p_drop=0.0),
    },
}


@torch.no_grad()
def predict(
    model: RAMEIFusion,
    store: FeatureStore,
    idx: np.ndarray,
    variants: dict[str, list[str | None]],
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    keys = ("logits", "branch_logits", "reliability", "weights")
    out: dict[str, list[np.ndarray]] = {k: [] for k in (*keys, "avail")}
    for s in range(0, len(idx), batch_size):
        sl = slice(s, s + batch_size)
        batch = store.batch(idx[sl], {m: v[sl] for m, v in variants.items()}, DEVICE)
        res = model(batch)
        for k in keys:
            out[k].append(res[k].float().cpu().numpy())
        out["avail"].append(batch["avail"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _fixed_mixed(n: int, seed: int) -> dict[str, list[str | None]]:
    return FeatureStore.random_variants(n, np.random.default_rng(seed), TrainSampling())


def train_fold(
    store: FeatureStore,
    fm: pd.DataFrame,
    exp: dict,
    seed: int,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[RAMEIFusion, float, list]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = {s: np.flatnonzero((fm["split"] == s).to_numpy()) for s in ("train", "val", "test")}
    store.fit_quality_stats(idx["train"])
    model = RAMEIFusion(FusionConfig(use_reliability=exp["use_reliability"])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(idx["train"]) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    cw = class_weights(store.y[idx["train"]], NUM_CLASSES).to(DEVICE)
    val_sets = {
        "clean": store.scenario_variants(len(idx["val"]), "clean"),
        "mixed": _fixed_mixed(len(idx["val"]), seed + 1000),
    }
    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(idx["train"])
        for s in range(0, len(order), batch_size):
            b_idx = order[s : s + batch_size]
            variants = FeatureStore.random_variants(len(b_idx), rng, exp["sampling"])
            batch = store.batch(b_idx, variants, DEVICE)
            y = torch.as_tensor(store.y[b_idx], device=DEVICE)
            loss = fusion_loss(model(batch), y, batch["avail"], cw)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        scores = {}
        for name, v in val_sets.items():
            p = predict(model, store, idx["val"], v)["logits"].argmax(1)
            scores[name] = f1_score(store.y[idx["val"]], p, average="macro")
        score = float(np.mean(list(scores.values())))
        history.append({"epoch": epoch, **scores, "score": score})
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_logits = np.concatenate(
        [predict(model, store, idx["val"], v)["logits"] for v in val_sets.values()]
    )
    temperature = fit_temperature(val_logits, np.concatenate([store.y[idx["val"]]] * 2))
    return model, temperature, history


def evaluate(
    model: RAMEIFusion,
    store: FeatureStore,
    test_idx: np.ndarray,
    temperature: float,
    with_baselines: bool,
) -> dict[str, Any]:
    y = store.y[test_idx]
    results = {}
    for scenario in SCENARIOS:
        r = predict(model, store, test_idx, store.scenario_variants(len(test_idx), scenario))
        fused = softmax(r["logits"], temperature)
        rep = classification_report(y, fused, LABELS)
        entry = {
            "macro_f1": rep["macro_f1"],
            "wa": rep["weighted_accuracy"],
            "ece": rep["ece"],
            "abstain_rate_at_0.45": float((fused.max(1) < 0.45).mean()),
            "reliability": dict(
                zip(MODALITIES, r["reliability"].mean(0).round(4).tolist(), strict=True)
            ),
            "weights": dict(zip(MODALITIES, r["weights"].mean(0).round(4).tolist(), strict=True)),
        }
        if with_baselines:
            probs = softmax(r["branch_logits"].reshape(-1, NUM_CLASSES)).reshape(
                r["branch_logits"].shape
            )
            avail = r["avail"][..., None]
            static = (probs * avail).sum(1) / avail.sum(1).clip(min=1)
            rel = r["reliability"][..., None] * avail
            rel_late = (probs * rel).sum(1) / rel.sum(1).clip(min=1e-6)
            entry["e5_static"] = float(f1_score(y, static.argmax(1), average="macro"))
            entry["e6_rel_late"] = float(f1_score(y, rel_late.argmax(1), average="macro"))
            for k, m in enumerate(MODALITIES):
                has = r["avail"][:, k].astype(bool)
                if has.any():  # branch-only score on the clips where that modality exists
                    entry[f"branch_{m}"] = float(
                        f1_score(y[has], probs[has, k].argmax(1), average="macro")
                    )
                    entry[f"coverage_{m}"] = float(has.mean())
        results[scenario] = entry
    return results


def aggregate(folds: list[dict]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        # Union of metric keys: some (e.g. branch-only scores) can be absent in a fold.
        keys = sorted({k for f in folds for k, v in f[scenario].items() if isinstance(v, float)})
        summary[scenario] = {}
        for k in keys:
            vals = [f[scenario][k] for f in folds if k in f[scenario]]
            summary[scenario][k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n_folds": len(vals),
            }
        for k in ("reliability", "weights"):
            summary[scenario][k] = {
                m: round(float(np.mean([f[scenario][k][m] for f in folds])), 4) for m in MODALITIES
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = FeatureStore()
    print(f"{len(store.clips)} clips, {store.clips['speaker_id'].nunique()} speakers", flush=True)
    fold_of = speaker_folds(store.clips, args.folds, args.seed)
    out_dir = (
        ARTIFACTS_DIR
        / "experiments"
        / "m2_fusion_crema_d"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name in args.experiments:
        exp = EXPERIMENTS[name]
        folds = []
        for f in range(args.folds):
            t0 = time.time()
            fm = fold_manifest(store.clips, fold_of, f, args.folds)
            model, temperature, history = train_fold(store, fm, exp, args.seed + f, args.epochs)
            test_idx = np.flatnonzero((fm["split"] == "test").to_numpy())
            res = evaluate(model, store, test_idx, temperature, with_baselines=name == "e8_rgcmt")
            res["_temperature"], res["_epochs_best"] = (
                temperature,
                int(np.argmax([h["score"] for h in history]) + 1),
            )
            folds.append(res)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "temperature": temperature,
                    "q_stats": store.q_stats,
                    "use_reliability": exp["use_reliability"],
                },
                out_dir / f"{name}_fold{f}.pt",
            )
            c = res["clean"]
            print(
                f"{name} fold {f}: clean F1={c['macro_f1']:.4f} ECE={c['ece']:.4f} "
                f"noisy_room F1={res['noisy_room_0db']['macro_f1']:.4f} "
                f"camera_off F1={res['camera_off']['macro_f1']:.4f} "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )
        all_results[name] = {
            "summary": aggregate(
                [{k: v for k, v in f.items() if not k.startswith("_")} for f in folds]
            ),
            "folds": folds,
        }
        (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"Results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
