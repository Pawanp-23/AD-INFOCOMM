"""Stage S1 — text branch on MELD (TRD §4.2).

MELD (Friends dialogues) is natural conversational text, unlike the fixed sentences of acted
corpora. Official train/dev/test splits are used; model selection on dev macro-F1, temperature
fitted on dev, metrics reported on test.

Run from ml/:
    python -m ramei_ml.training.text_branch
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoTokenizer

from ramei_ml.data.labels import EMOTION_TO_ID, EMOTIONS, NUM_CLASSES, map_label
from ramei_ml.evaluation.calibration import fit_temperature, softmax
from ramei_ml.evaluation.metrics import classification_report
from ramei_ml.models.text_branch import TextBranch, TextBranchConfig
from ramei_ml.paths import ARTIFACTS_DIR, RAW_DIR
from ramei_ml.training.torch_trainer import class_weights

MELD_DIR = RAW_DIR / "meld"
MODEL_DIR = ARTIFACTS_DIR / "models" / "text_branch"


def load_meld(meld_dir: Path = MELD_DIR) -> pd.DataFrame:
    frames = []
    for split, name in (("train", "train"), ("val", "dev"), ("test", "test")):
        df = pd.read_csv(meld_dir / f"{name}_sent_emo.csv", encoding="utf-8")
        frames.append(
            pd.DataFrame(
                {
                    "text": df["Utterance"].astype(str).str.replace("\u0092", "'").str.strip(),
                    "emotion": df["Emotion"].map(lambda e: map_label("meld", e)),
                    "speaker": df["Speaker"],
                    "split": split,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out[out["emotion"].notna()].reset_index(drop=True)


def batches(
    texts: list[str],
    labels: np.ndarray | None,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int = 0,
):
    order = (
        np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    )
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        enc = tokenizer(
            [texts[j] for j in idx],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield enc, (torch.as_tensor(labels[idx]) if labels is not None else None)


@torch.no_grad()
def encode(
    model: TextBranch, tokenizer, texts: list[str], device: str, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """(z [N, 768], logits [N, 7]) for a list of texts. Empty strings are encoded as-is."""
    model.eval()
    zs, logits = [], []
    for enc, _ in batches(texts, None, tokenizer, batch_size, model.cfg.max_length, False):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        zs.append(out["z"].float().cpu().numpy())
        logits.append(out["logits"].float().cpu().numpy())
    return np.concatenate(zs), np.concatenate(logits)


def train(epochs: int = 4, batch_size: int = 16, lr: float = 3e-4, seed: int = 42) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    data = load_meld()
    cfg = TextBranchConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained)
    model = TextBranch(cfg).to(device)
    split = {s: data[data["split"] == s] for s in ("train", "val", "test")}
    texts = {s: d["text"].tolist() for s, d in split.items()}
    y = {s: d["emotion"].map(EMOTION_TO_ID).to_numpy() for s, d in split.items()}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(texts["train"]) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y["train"], NUM_CLASSES).to(device), label_smoothing=0.1
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for enc, labels in batches(
            texts["train"], y["train"], tokenizer, batch_size, cfg.max_length, True, seed + epoch
        ):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))[
                    "logits"
                ]
                loss = criterion(logits.float(), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(labels)
            seen += len(labels)
        _, val_logits = encode(model, tokenizer, texts["val"], device)
        val_f1 = float(f1_score(y["val"], val_logits.argmax(1), average="macro"))
        print(f"epoch {epoch}: loss={total / seen:.4f} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, model.trainable_state_dict()

    model.load_state_dict(best_state, strict=False)
    _, val_logits = encode(model, tokenizer, texts["val"], device)
    _, test_logits = encode(model, tokenizer, texts["test"], device)
    temperature = fit_temperature(val_logits, y["val"])
    labels = list(EMOTIONS)
    report = classification_report(y["test"], softmax(test_logits, temperature), labels)
    report["ece_raw"] = classification_report(y["test"], softmax(test_logits), labels)["ece"]
    report["weighted_f1"] = float(f1_score(y["test"], test_logits.argmax(1), average="weighted"))
    result = {
        "dataset": "MELD",
        "best_val_macro_f1": best_f1,
        "temperature": temperature,
        "test": report,
        "sizes": {s: len(v) for s, v in texts.items()},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / "trainable_state.pt")
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
    t = report
    print(
        f"TEST macro_f1={t['macro_f1']:.4f} weighted_f1={t['weighted_f1']:.4f} "
        f"wa={t['weighted_accuracy']:.4f} ece_raw={t['ece_raw']:.4f} ece={t['ece']:.4f} "
        f"T={temperature:.2f}"
    )
    return result


def load_trained(
    device: str = "cpu", model_dir: Path = MODEL_DIR
) -> tuple[TextBranch, object, float]:
    """Load the trained text branch, its tokenizer and the fitted temperature."""
    cfg = TextBranchConfig(**json.loads((model_dir / "config.json").read_text()))
    model = TextBranch(cfg)
    model.load_state_dict(torch.load(model_dir / "trainable_state.pt"), strict=False)
    temperature = json.loads((model_dir / "metrics.json").read_text())["temperature"]
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg.pretrained), temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
