"""Build the multimodal feature store for M2 fusion (TRD §3.5, §4.1-4.4).

Every clip is passed once through frozen encoders, both clean and corrupted:
  audio  : WavLM-Base+ per-layer mean/std pooled states [13, 1536]  (clean, snr10, snr0)
  text   : Whisper transcript -> MELD-trained DeBERTa z_t [768] + logits [7]
           (asr_clean, asr_snr0 = ASR on noisy audio, drop50 = 50% word deletions)
  visual : ViT face-expression per-frame CLS [16, 768] + per-frame logits [16, 7]
           (clean, blur, dark, occlude)
plus quality features per modality. Corrupted variants let fusion learn when to distrust a
modality without leaking test speakers (encoders are frozen, nothing is fitted here).

Run from ml/ (each step is resumable):
    python -m ramei_ml.fusion.extract audio
    python -m ramei_ml.fusion.extract asr
    python -m ramei_ml.fusion.extract text
    python -m ramei_ml.fusion.extract visual
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ramei_ml.data.corruptions import add_noise, blur_faces, darken_faces, drop_words, occlude_faces
from ramei_ml.features.audio import AudioConfig, load_audio
from ramei_ml.features.quality import (
    text_quality,
)
from ramei_ml.fusion.encoders import AudioEncoder, FaceEncoder
from ramei_ml.paths import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR

STORE = PROCESSED_DIR / "fusion" / "crema_d"
FACES = PROCESSED_DIR / "faces" / "crema_d"
AUDIO_VARIANTS = {"clean": None, "snr10": 10.0, "snr0": 0.0}
VISUAL_VARIANTS = ("clean", "blur", "dark", "occlude")
TEXT_VARIANTS = ("asr_clean", "asr_snr0", "drop50")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIT = "trpakov/vit-face-expression"


def clip_table() -> pd.DataFrame:
    """CREMA-D clips that have both audio and face crops, with speaker/gender/emotion."""
    manifest = pd.read_csv(MANIFEST_DIR / "ravdess_crema_audio.csv")
    df = manifest[manifest["dataset"] == "crema_d"].copy()
    df["stem"] = df["path"].map(lambda p: Path(p).stem)
    df = df[df["stem"].map(lambda s: (FACES / s / "quality.json").exists())]
    return df.drop(columns=["split"]).sort_values("stem").reset_index(drop=True)


def noisy(y: np.ndarray, snr: float | None, seed: int) -> np.ndarray:
    return y if snr is None else add_noise(y, snr, np.random.default_rng(seed))


def _save(name: str, **arrays: np.ndarray) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    np.savez(STORE / f"{name}.npz", **arrays)


# ----------------------------------------------------------------------------- audio
def extract_audio(clips: pd.DataFrame) -> None:
    encoder = AudioEncoder(DEVICE)
    cfg = AudioConfig()
    for variant, snr in AUDIO_VARIANTS.items():
        if (STORE / f"audio_{variant}.npz").exists():
            continue
        feats, quality = [], []
        for i, path in enumerate(clips["path"]):
            f, q = encoder(noisy(load_audio(path, cfg), snr, seed=i))
            feats.append(f)
            quality.append(q)
            if (i + 1) % 1000 == 0:
                print(f"  audio/{variant}: {i + 1}/{len(clips)}", flush=True)
        _save(
            f"audio_{variant}", feats=np.stack(feats).astype(np.float16), quality=np.stack(quality)
        )
        print(f"saved audio_{variant}", flush=True)


# ----------------------------------------------------------------------------- ASR
def extract_asr(clips: pd.DataFrame) -> None:
    from ramei_ml.features.asr import Transcriber

    out = STORE / "transcripts.csv"
    transcriber = Transcriber(
        "small.en", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8"
    )
    rows = []
    for i, path in enumerate(clips["path"]):
        y = load_audio(path, AudioConfig())
        for variant, snr in (("asr_clean", None), ("asr_snr0", 0.0)):
            t = transcriber.transcribe(noisy(y, snr, seed=i))
            rows.append(
                {
                    "stem": clips["stem"].iloc[i],
                    "variant": variant,
                    "text": t.text,
                    "avg_logprob": t.avg_logprob,
                    "no_speech_prob": t.no_speech_prob,
                    "compression_ratio": t.compression_ratio,
                    "n_words": t.n_words,
                }
            )
        if (i + 1) % 500 == 0:
            print(f"  asr: {i + 1}/{len(clips)}", flush=True)
    STORE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}", flush=True)


# ----------------------------------------------------------------------------- text
def extract_text(clips: pd.DataFrame) -> None:
    from ramei_ml.training.text_branch import encode, load_trained

    model, tokenizer, _ = load_trained(DEVICE)
    tr = pd.read_csv(STORE / "transcripts.csv", keep_default_na=False)
    for variant in TEXT_VARIANTS:
        source = "asr_clean" if variant == "drop50" else variant
        t = tr[tr["variant"] == source].set_index("stem").loc[clips["stem"]].reset_index()
        texts = t["text"].astype(str).tolist()
        if variant == "drop50":
            rng = np.random.default_rng(0)
            texts = [drop_words(x, 0.5, rng) for x in texts]
        n_words = [len(x.split()) for x in texts]
        z, logits = encode(model, tokenizer, texts, DEVICE)
        quality = np.stack(
            [
                text_quality(a, b, c, n)
                for a, b, c, n in zip(
                    t["avg_logprob"],
                    t["no_speech_prob"],
                    t["compression_ratio"],
                    n_words,
                    strict=True,
                )
            ]
        )
        if variant == "drop50":  # deletions make ASR output look less confident
            quality[:, 0] -= 0.5
        _save(
            f"text_{variant}",
            z=z.astype(np.float16),
            logits=logits,
            quality=quality,
            texts=np.array(texts),
        )
        print(f"saved text_{variant}", flush=True)


# ----------------------------------------------------------------------------- visual
def extract_visual(clips: pd.DataFrame) -> None:
    from ramei_ml.features.video import load_faces

    encoder = FaceEncoder(DEVICE)
    for variant in VISUAL_VARIANTS:
        if (STORE / f"visual_{variant}.npz").exists():
            continue
        cls, logits, masks, quality = [], [], [], []
        rng = np.random.default_rng(0)
        for i, stem in enumerate(clips["stem"]):
            crops, q = load_faces(FACES / stem)
            present = np.array(q["face_present"]) > 0
            if variant == "blur":
                crops = blur_faces(crops)
            elif variant == "dark":
                crops = darken_faces(crops, rng=rng)
            elif variant == "occlude":
                crops = occlude_faces(crops, rng)
            c, lg, qv = encoder(crops, present, np.array(q["det_conf"]), np.array(q["face_area"]))
            cls.append(c)
            logits.append(lg)
            masks.append(present)
            quality.append(qv)
            if (i + 1) % 1000 == 0:
                print(f"  visual/{variant}: {i + 1}/{len(clips)}", flush=True)
        _save(
            f"visual_{variant}",
            cls=np.stack(cls).astype(np.float16),
            logits=np.stack(logits),
            mask=np.stack(masks),
            quality=np.stack(quality),
        )
        print(f"saved visual_{variant}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["audio", "asr", "text", "visual", "index"])
    args = parser.parse_args()
    clips = clip_table()
    print(f"{len(clips)} CREMA-D clips with audio + faces", flush=True)
    if args.step == "index" or not (STORE / "clips.csv").exists():
        STORE.mkdir(parents=True, exist_ok=True)
        clips.to_csv(STORE / "clips.csv", index=False)
    else:  # every step must use the identical clip order
        saved = pd.read_csv(STORE / "clips.csv")
        if saved["stem"].tolist() != clips["stem"].tolist():
            raise SystemExit("Clip set changed since 'index' — delete the store and rebuild.")
    {
        "audio": extract_audio,
        "asr": extract_asr,
        "text": extract_text,
        "visual": extract_visual,
        "index": lambda c: None,
    }[args.step](clips)
    (STORE / "README.json").write_text(
        json.dumps(
            {
                "audio_variants": list(AUDIO_VARIANTS),
                "visual_variants": list(VISUAL_VARIANTS),
                "text_variants": list(TEXT_VARIANTS),
                "raw_dir": str(RAW_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
