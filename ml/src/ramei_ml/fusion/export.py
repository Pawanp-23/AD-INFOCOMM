"""Build the deployable RAMEI model bundle consumed by the backend (SAD §6.4).

The final fusion model is trained on all CREMA-D speakers except a gender-balanced validation
group (~1/7 of speakers) used for early stopping and temperature scaling. Reported quality
numbers come from the cross-validation run (fusion/train.py), not from this model.

Run from ml/:
    python -m ramei_ml.fusion.export --version 1.0.0 [--cv-results <results.json>]
"""

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from ramei_ml.data.labels import EMOTIONS
from ramei_ml.fusion.data import FeatureStore
from ramei_ml.fusion.encoders import VIT, WAVLM
from ramei_ml.fusion.train import EXPERIMENTS, train_fold
from ramei_ml.paths import ARTIFACTS_DIR, ML_ROOT
from ramei_ml.training.cv import fold_manifest, speaker_folds
from ramei_ml.training.text_branch import MODEL_DIR as TEXT_MODEL_DIR

BUNDLE_ROOT = ARTIFACTS_DIR / "bundles"
THRESHOLDS = {
    "abstain_confidence": 0.45,
    "abstain_reliability": 0.30,
    "conflict_reliability": 0.60,
    "conflict_js": 0.50,
}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ML_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_card(version: str, cv: dict | None) -> str:
    lines = [
        f"# RAMEI model card — ramei-rgcmt {version}",
        "",
        "**Task:** 7-class emotion recognition (neutral, happy, sad, angry, fear, disgust, "
        "surprise) from speech audio, transcript and face video, with per-modality "
        "reliability scores and calibrated confidence.",
        "",
        "## Components",
        f"- Audio: frozen `{WAVLM}`, per-layer mean/std pooling + trained head",
        "- Text: `faster-whisper small.en` ASR -> `microsoft/deberta-v3-base` + LoRA "
        "trained on MELD",
        f"- Visual: MediaPipe face detector -> frozen `{VIT}` per frame + temporal head",
        "- Fusion: Reliability-Gated Cross-Modal Transformer (ADR-0001), temperature-scaled",
        "",
        "## Training data",
        "- CREMA-D (91 actors, 7,442 clips; ODbL) for fusion; MELD (Friends dialogues) for "
        "the text encoder.",
        "- Corrupted variants: pink noise at 10 / 0 dB SNR, ASR on noisy audio, 50% word "
        "deletions, blurred / darkened / half-occluded faces, random modality dropout.",
        "",
        "## Evaluation (speaker-independent cross-validation)",
    ]
    if cv:
        lines += ["| Scenario | Macro-F1 | ECE |", "|---|---|---|"]
        for scenario, m in cv.items():
            lines.append(
                f"| {scenario} | {m['macro_f1']['mean']:.3f} ± {m['macro_f1']['std']:.3f}"
                f" | {m['ece']['mean']:.3f} |"
            )
    else:
        lines.append("Not attached — see docs/reports/M2 report.")
    lines += [
        "",
        "## Limitations",
        "- Acted English speech (CREMA-D); CREMA-D actors read 12 fixed sentences, so the "
        "transcript carries little emotional information on this data.",
        "- Not validated on spontaneous speech, children, other languages or clinical use.",
        "- Valence/arousal are estimated from category probabilities, not trained regressors.",
        "- Assistive use only; never for automated decisions about people.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--cv-results", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = FeatureStore()
    fold_of = speaker_folds(store.clips, 7, args.seed)
    fm = fold_manifest(store.clips, fold_of, 0, 7)  # fold 0 -> "test", fold 1 -> "val"
    fm.loc[fm["split"] == "test", "split"] = "train"  # no test set: use everything but val
    exp = EXPERIMENTS["e8_rgcmt"]
    model, temperature, history = train_fold(store, fm, exp, args.seed, args.epochs)

    name = f"ramei-rgcmt-{args.version}"
    out = BUNDLE_ROOT / name
    if out.exists():
        shutil.rmtree(out)
    (out / "text_branch").mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.c),
            "temperature": temperature,
            "q_stats": {m: [s[0].tolist(), s[1].tolist()] for m, s in store.q_stats.items()},
        },
        out / "fusion.pt",
    )
    for f in ("trainable_state.pt", "config.json", "metrics.json"):
        shutil.copy(TEXT_MODEL_DIR / f, out / "text_branch" / f)

    cv = None
    if args.cv_results and args.cv_results.exists():
        cv = json.loads(args.cv_results.read_text())["e8_rgcmt"]["summary"]
    manifest = {
        "schema_version": 1,
        "name": name,
        "version": args.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "labels": list(EMOTIONS),
        "encoders": {
            "audio": WAVLM,
            "visual": VIT,
            "asr": "small.en",
            "text": "microsoft/deberta-v3-base + text_branch/",
        },
        "thresholds": THRESHOLDS,
        "temperature": temperature,
        "train": {
            "dataset": "crema_d",
            "speakers": int(fm["speaker_id"].nunique()),
            "val_speakers": int(fm.loc[fm["split"] == "val", "speaker_id"].nunique()),
            "best_epoch": int(max(history, key=lambda h: h["score"])["epoch"]),
            "best_val_score": round(max(h["score"] for h in history), 4),
        },
        "cv_summary": cv,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "labels.json").write_text(json.dumps(list(EMOTIONS)))
    (out / "MODEL_CARD.md").write_text(model_card(args.version, cv), encoding="utf-8")
    print(f"Bundle written: {out}")


if __name__ == "__main__":
    main()
