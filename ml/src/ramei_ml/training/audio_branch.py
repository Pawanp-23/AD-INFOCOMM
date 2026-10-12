"""Stage S1 — audio branch training with speaker-independent cross-validation (TRD §4.1, §5).

Run from ml/:
    python -m ramei_ml.training.audio_branch                 # E2 LoRA, RAVDESS, 6 folds
    python -m ramei_ml.training.audio_branch experiment=e2_audio_frozen
    python -m ramei_ml.training.audio_branch data=ravdess_crema cv_folds=4

Per fold: train on train speakers, early-stop on validation macro-F1, fit temperature on
validation logits, report raw and calibrated metrics on the held-out test speakers.
"""

import json
import math
import time
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from ramei_ml.data.audio_dataset import (
    AugmentConfig,
    WaveformDataset,
    build_waveform_cache,
    collate_waveforms,
    seed_worker,
)
from ramei_ml.data.labels import EMOTION_TO_ID, EMOTIONS, NUM_CLASSES
from ramei_ml.data.manifest import assert_speaker_independent
from ramei_ml.evaluation.calibration import fit_temperature, softmax
from ramei_ml.evaluation.metrics import classification_report, group_report
from ramei_ml.models.audio_branch import AudioBranch, AudioBranchConfig
from ramei_ml.paths import ARTIFACTS_DIR, PROCESSED_DIR, resolve
from ramei_ml.training.cv import fold_manifest, speaker_folds
from ramei_ml.training.torch_trainer import class_weights

LABELS = list(EMOTIONS)
SUMMARY_METRICS = ("weighted_accuracy", "unweighted_accuracy", "macro_f1", "ece_raw", "ece")


def _loader(
    ds: WaveformDataset, batch_size: int, shuffle: bool, workers: int, seed: int
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_waveforms,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def predict_logits(model: AudioBranch, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for waves, lengths, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(waves.to(device), lengths.to(device))["logits"]
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def train_fold(
    cfg: DictConfig, manifest: pd.DataFrame, cache_files: list, fold_seed: int
) -> dict[str, Any]:
    t = cfg.experiment.train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(fold_seed)
    sets = {}
    for split in ("train", "val", "test"):
        idx = np.flatnonzero((manifest["split"] == split).to_numpy())
        sets[split] = (manifest.iloc[idx], [cache_files[i] for i in idx])
    augment = AugmentConfig(**OmegaConf.to_container(t.augment))
    train_ds = WaveformDataset(*sets["train"], train=True, augment=augment, seed=fold_seed)
    loaders = {
        "train": _loader(train_ds, t.batch_size, True, t.num_workers, fold_seed),
        **{
            s: _loader(WaveformDataset(*sets[s]), t.eval_batch_size, False, t.num_workers, 0)
            for s in ("val", "test")
        },
    }
    y = {s: sets[s][0]["emotion"].map(EMOTION_TO_ID).to_numpy() for s in sets}

    model = AudioBranch(AudioBranchConfig(**OmegaConf.to_container(cfg.experiment.model))).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameter_groups(t.lr_encoder, t.lr_head), weight_decay=t.weight_decay
    )
    steps_per_epoch = math.ceil(len(loaders["train"]) / t.grad_accum)
    total_steps = steps_per_epoch * t.epochs
    warmup = int(t.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            (s + 1) / max(1, warmup)
            if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
        ),
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y["train"], NUM_CLASSES).to(device), label_smoothing=t.label_smoothing
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_state, stale, history = -1.0, None, 0, []
    started = time.time()
    for epoch in range(1, t.epochs + 1):
        model.train()
        if model.cfg.mode == "frozen":
            model.encoder.eval()  # no dropout / SpecAugment inside a frozen encoder
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (waves, lengths, labels) in enumerate(loaders["train"], start=1):
            waves, lengths, labels = waves.to(device), lengths.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = criterion(model(waves, lengths)["logits"].float(), labels)
            scaler.scale(loss / t.grad_accum).backward()
            if step % t.grad_accum == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item() * len(labels)
            seen += len(labels)

        val_logits = predict_logits(model, loaders["val"], device)
        val_f1 = float(f1_score(y["val"], val_logits.argmax(1), average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / seen,
                "val_macro_f1": val_f1,
                "minutes": round((time.time() - started) / 60, 2),
            }
        )
        print(
            f"  epoch {epoch:2d}  loss={running / seen:.4f}  val_f1={val_f1:.4f}  "
            f"[{history[-1]['minutes']} min]",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, best_state, stale = val_f1, model.trainable_state_dict(), 0
        else:
            stale += 1
            if stale >= t.patience:
                break

    model.load_state_dict(best_state, strict=False)
    logits = {s: predict_logits(model, loaders[s], device) for s in ("val", "test")}
    temperature = fit_temperature(logits["val"], y["val"])
    probs_raw = softmax(logits["test"])
    probs_cal = softmax(logits["test"], temperature)
    test_report = classification_report(y["test"], probs_cal, LABELS)
    test_report["ece_raw"] = classification_report(y["test"], probs_raw, LABELS)["ece"]
    test_df = sets["test"][0].reset_index(drop=True)
    result = {
        "test": test_report,
        "temperature": temperature,
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
        "layer_weights": model.layer_weights.softmax(0).tolist(),
        "test_by_gender": group_report(y["test"], probs_cal, test_df["gender"], LABELS),
        "test_by_dataset": group_report(y["test"], probs_cal, test_df["dataset"], LABELS),
        "history": history,
        "_state": best_state,
        "_test_logits": logits["test"],
        "_test_paths": test_df["path"].tolist(),
    }
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run(cfg: DictConfig) -> dict[str, Any]:
    manifest = pd.read_csv(resolve(cfg.data.manifest))
    assert_speaker_independent(manifest)
    cache_root = resolve(cfg.get("cache_dir") or PROCESSED_DIR / "wave16k")
    print(f"Caching {len(manifest)} waveforms -> {cache_root}")
    cache_files = build_waveform_cache(manifest["path"].tolist(), cache_root, n_jobs=cfg.n_jobs)

    k = int(cfg.cv_folds)
    if k > 1:
        fold_of = speaker_folds(manifest, k, cfg.seed)
        fold_manifests = [fold_manifest(manifest, fold_of, f, k) for f in range(k)]
    else:
        fold_manifests = [manifest]
    max_folds = cfg.get("max_folds") or len(fold_manifests)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_root = resolve(cfg.get("output_root") or ARTIFACTS_DIR / "experiments")
    tag = f"{cfg.data.name}_cv{k}" if k > 1 else cfg.data.name
    out_dir = exp_root / f"{cfg.experiment.name}__{tag}" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    folds = []
    for f, fm in enumerate(fold_manifests[:max_folds]):
        speakers = sorted(fm.loc[fm["split"] == "test", "speaker_id"].unique())
        print(f"Fold {f}: test speakers {len(speakers)}", flush=True)
        r = train_fold(cfg, fm, cache_files, cfg.seed + f)
        fold_dir = out_dir / f"fold{f}"
        fold_dir.mkdir()
        torch.save(r.pop("_state"), fold_dir / "trainable_state.pt")
        np.savez_compressed(
            fold_dir / "test_logits.npz",
            logits=r.pop("_test_logits"),
            paths=np.array(r.pop("_test_paths")),
        )
        pd.DataFrame(r.pop("history")).to_csv(fold_dir / "history.csv", index=False)
        (fold_dir / "metrics.json").write_text(json.dumps(r, indent=2))
        folds.append({"fold": f, "test_speakers": speakers, **r})
        t = r["test"]
        print(
            f"Fold {f}: macro_f1={t['macro_f1']:.4f} wa={t['weighted_accuracy']:.4f} "
            f"ece_raw={t['ece_raw']:.4f} ece_cal={t['ece']:.4f} T={r['temperature']:.2f}",
            flush=True,
        )

    summary = {
        m: {
            "mean": float(np.mean([f["test"][m] for f in folds])),
            "std": float(np.std([f["test"][m] for f in folds])),
        }
        for m in SUMMARY_METRICS
    }
    by_gender: dict[str, list[float]] = {}
    by_dataset: dict[str, list[float]] = {}
    for f in folds:
        for g, v in f["test_by_gender"].items():
            by_gender.setdefault(g, []).append(v["macro_f1"])
        for d, v in f["test_by_dataset"].items():
            by_dataset.setdefault(d, []).append(v["macro_f1"])
    result = {
        "experiment": cfg.experiment.name,
        "data": cfg.data.name,
        "folds_k": k,
        "folds_run": len(folds),
        "seed": cfg.seed,
        "summary": summary,
        "macro_f1_by_gender": {g: float(np.mean(v)) for g, v in by_gender.items()},
        "macro_f1_by_dataset": {d: float(np.mean(v)) for d, v in by_dataset.items()},
        "folds": [
            {k_: v for k_, v in f.items() if k_ not in ("test",)}
            | {"macro_f1": f["test"]["macro_f1"], "ece": f["test"]["ece"]}
            for f in folds
        ],
    }
    (out_dir / "cv_metrics.json").write_text(json.dumps(result, indent=2))

    row = {
        "timestamp": stamp,
        "experiment": cfg.experiment.name,
        "data": tag,
        "split": f"speaker_cv{k}" if k > 1 else "speaker",
        "test_wa": round(summary["weighted_accuracy"]["mean"], 4),
        "test_ua": round(summary["unweighted_accuracy"]["mean"], 4),
        "test_macro_f1": round(summary["macro_f1"]["mean"], 4),
        "test_ece": round(summary["ece"]["mean"], 4),
        "run_dir": out_dir.relative_to(exp_root).as_posix(),
    }
    board = exp_root / "leaderboard.csv"
    pd.DataFrame([row]).to_csv(board, mode="a", header=not board.exists(), index=False)

    print(
        "\nSUMMARY "
        + "  ".join(f"{m}={v['mean']:.4f}+/-{v['std']:.4f}" for m, v in summary.items())
    )
    print(f"Run directory: {out_dir}")
    return result


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="audio")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
