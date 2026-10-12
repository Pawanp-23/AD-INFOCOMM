"""Frozen per-modality encoders shared by training-time extraction and serving.

Training (fusion/extract.py) and the backend inference engine must compute identical features,
so both go through these classes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ramei_ml.data.labels import EMOTIONS
from ramei_ml.features.quality import (
    audio_quality,
    crops_blur_brightness,
    text_quality,
    visual_quality,
)

WAVLM = "microsoft/wavlm-base-plus"
VIT = "trpakov/vit-face-expression"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class AudioEncoder:
    """WavLM-Base+ per-layer mean/std pooled hidden states [13, 1536] + audio quality."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import WavLMModel

        self.device = device or default_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = WavLMModel.from_pretrained(WAVLM, dtype=torch.float32)
        self.model = self.model.to(self.device, self.dtype).eval()

    @torch.no_grad()
    def __call__(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """y: mono float32 16 kHz (already preprocessed). Returns (feats [13,1536], quality [5])."""
        if y.size < 16_000 // 4:  # WavLM needs a minimum length; pad very short input
            y = np.pad(y, (0, 16_000 // 4 - y.size))
        wave = torch.from_numpy(y.astype(np.float32))[None].to(self.device, self.dtype)
        hidden = torch.stack(self.model(wave, output_hidden_states=True).hidden_states)[:, 0]
        feats = torch.cat([hidden.mean(1), hidden.std(1)], dim=-1).float().cpu().numpy()
        return feats, audio_quality(y)


@dataclass
class AsrSignals:
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class TextEncoder:
    """MELD-trained DeBERTa text branch: (z [768], logits [7], quality [5])."""

    def __init__(self, device: str | None = None, model_dir: Path | None = None) -> None:
        from ramei_ml.training.text_branch import MODEL_DIR, load_trained

        self.device = device or default_device()
        self.model, self.tokenizer, self.temperature = load_trained(
            self.device, model_dir or MODEL_DIR
        )

    def __call__(
        self, texts: list[str], asr: list[AsrSignals]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ramei_ml.training.text_branch import encode

        z, logits = encode(self.model, self.tokenizer, texts, self.device)
        quality = np.stack(
            [
                text_quality(a.avg_logprob, a.no_speech_prob, a.compression_ratio, len(t.split()))
                for t, a in zip(texts, asr, strict=True)
            ]
        )
        return z, logits, quality


class FaceEncoder:
    """ViT face-expression encoder over face crops: (cls [N,768], logits [N,7], quality [6])."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import ViTForImageClassification

        self.device = device or default_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = ViTForImageClassification.from_pretrained(VIT, dtype=torch.float32)
        self.model = self.model.to(self.device, self.dtype).eval()
        vit_labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.order = [vit_labels.index(e) for e in EMOTIONS]  # ViT label order -> RAMEI order

    @torch.no_grad()
    def __call__(
        self, crops: np.ndarray, present: np.ndarray, det_conf: np.ndarray, face_area: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """crops uint8 [N, 224, 224, 3] RGB; present/det_conf/face_area per frame [N]."""
        blur, bright = crops_blur_brightness(crops)
        quality = visual_quality(present.astype(float), det_conf, face_area, blur, bright)
        x = torch.from_numpy(crops).to(self.device).permute(0, 3, 1, 2).to(self.dtype)
        out = self.model(pixel_values=x / 127.5 - 1.0, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0].float().cpu().numpy()
        logits = out.logits.float().cpu().numpy()[:, self.order]
        return cls, logits, quality
