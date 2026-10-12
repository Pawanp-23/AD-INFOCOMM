"""Inference engine: loads the versioned model bundle once and analyzes segments (TRD §4, §7.1).

Per segment: audio -> WavLM features, transcript -> text branch, face crops -> ViT, then the
reliability-gated fusion model, temperature calibration, abstention, conflict detection and
human-readable explanations. All feature code is shared with training via ramei_ml.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.core.config import get_settings

LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
MODALITIES = ("audio", "text", "visual")
# Circumplex anchors (valence, arousal) per category; V/A = probability-weighted average.
VA_ANCHORS = {
    "neutral": (0.0, 0.0),
    "happy": (0.8, 0.5),
    "sad": (-0.7, -0.4),
    "angry": (-0.6, 0.8),
    "fear": (-0.6, 0.6),
    "disgust": (-0.6, 0.3),
    "surprise": (0.3, 0.8),
}


@dataclass
class SegmentInput:
    audio: np.ndarray | None  # mono float32 16 kHz (preprocessed), None if unavailable
    transcript: str | None
    asr_avg_logprob: float = -1.0
    asr_no_speech_prob: float = 0.0
    asr_compression_ratio: float = 1.0
    face_crops: np.ndarray | None = None  # uint8 [16, 224, 224, 3]
    face_quality: dict | None = None  # per-frame lists: face_present, det_conf, face_area


class ModelNotAvailableError(RuntimeError):
    pass


class InferenceEngine:
    _instance: "InferenceEngine | None" = None
    _lock = threading.Lock()

    def __init__(self, bundle_dir: Path, device: str) -> None:
        from ramei_ml.features.asr import Transcriber
        from ramei_ml.features.video import FaceDetector
        from ramei_ml.fusion.encoders import AudioEncoder, FaceEncoder, TextEncoder
        from ramei_ml.fusion.model import FusionConfig, RAMEIFusion

        self.bundle_dir = bundle_dir
        self.manifest = json.loads((bundle_dir / "manifest.json").read_text())
        self.version = self.manifest["name"]
        self.thresholds = self.manifest["thresholds"]
        self.device = device
        ckpt = torch.load(bundle_dir / "fusion.pt", map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        cfg["quality_dims"] = tuple(cfg["quality_dims"])
        self.fusion = RAMEIFusion(FusionConfig(**cfg))
        self.fusion.load_state_dict(ckpt["state_dict"])
        self.fusion.to(device).eval()
        self.temperature = float(ckpt["temperature"])
        self.q_stats = {m: (np.array(v[0]), np.array(v[1])) for m, v in ckpt["q_stats"].items()}
        self.audio_encoder = AudioEncoder(device)
        self.text_encoder = TextEncoder(device, bundle_dir / "text_branch")
        self.face_encoder = FaceEncoder(device)
        self.face_detector = FaceDetector()
        self.transcriber = Transcriber(
            "small.en", device=device, compute_type="float16" if device == "cuda" else "int8"
        )
        self.lock = threading.Lock()  # one GPU job at a time
        self.detector_lock = threading.Lock()  # MediaPipe detector is not thread-safe

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def get(cls) -> "InferenceEngine":
        with cls._lock:
            if cls._instance is None:
                settings = get_settings()
                bundle = settings.model_bundle_dir / settings.active_model
                if not (bundle / "manifest.json").exists():
                    raise ModelNotAvailableError(f"Model bundle not found: {bundle}")
                device = settings.inference_device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._instance = cls(bundle, device)
            return cls._instance

    @classmethod
    def status(cls) -> dict[str, Any]:
        settings = get_settings()
        bundle = settings.model_bundle_dir / settings.active_model
        manifest = None
        if (bundle / "manifest.json").exists():
            manifest = json.loads((bundle / "manifest.json").read_text())
        return {
            "active_model": settings.active_model,
            "bundle_found": manifest is not None,
            "loaded": cls._instance is not None,
            "device": cls._instance.device if cls._instance else None,
            "manifest": manifest,
        }

    # ------------------------------------------------------------------ ASR
    def transcribe_words(self, audio: np.ndarray) -> list[dict[str, Any]]:
        """Word-level transcript of a whole recording with Whisper confidence signals."""
        with self.lock:
            segments, _ = self.transcriber.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            words = []
            for s in segments:
                for w in s.words or []:
                    words.append(
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word.strip(),
                            "prob": w.probability,
                            "seg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                            "compression_ratio": s.compression_ratio,
                        }
                    )
        return words

    def transcribe_segment(self, audio: np.ndarray) -> tuple[str, float, float, float]:
        with self.lock:
            t = self.transcriber.transcribe(audio)
        return t.text, t.avg_logprob, t.no_speech_prob, t.compression_ratio

    # ------------------------------------------------------------------ analysis
    def _norm_q(self, m: str, q: np.ndarray) -> np.ndarray:
        mean, std = self.q_stats[m]
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=-10.0)
        return ((q - mean) / std).astype(np.float32)

    @torch.no_grad()
    def analyze(self, seg: SegmentInput) -> dict[str, Any]:
        from ramei_ml.fusion.encoders import AsrSignals

        with self.lock:
            batch = {
                "audio": np.zeros((1, 13, 1536), np.float32),
                "q_audio": np.zeros((1, 5), np.float32),
                "text_z": np.zeros((1, 768), np.float32),
                "text_logits": np.zeros((1, 7), np.float32),
                "q_text": np.zeros((1, 5), np.float32),
                "visual": np.zeros((1, 16, 768), np.float32),
                "visual_mask": np.zeros((1, 16), bool),
                "q_visual": np.zeros((1, 6), np.float32),
                "avail": np.zeros((1, 3), np.float32),
            }
            raw_q: dict[str, np.ndarray] = {}
            if seg.audio is not None and seg.audio.size >= 1600:
                feats, q = self.audio_encoder(seg.audio)
                batch["audio"][0], batch["q_audio"][0] = feats, self._norm_q("audio", q)
                batch["avail"][0, 0], raw_q["audio"] = 1, q
            if seg.transcript is not None:
                asr = AsrSignals(
                    seg.asr_avg_logprob, seg.asr_no_speech_prob, seg.asr_compression_ratio
                )
                z, logits, q = self.text_encoder([seg.transcript], [asr])
                batch["text_z"][0], batch["text_logits"][0] = z[0], logits[0]
                batch["q_text"][0] = self._norm_q("text", q[0])
                batch["avail"][0, 1], raw_q["text"] = 1, q[0]
            if seg.face_crops is not None and seg.face_quality is not None:
                present = np.array(seg.face_quality["face_present"]) > 0
                if present.any():
                    cls, _, q = self.face_encoder(
                        seg.face_crops,
                        present,
                        np.array(seg.face_quality["det_conf"]),
                        np.array(seg.face_quality["face_area"]),
                    )
                    batch["visual"][0], batch["visual_mask"][0] = cls, present
                    batch["q_visual"][0] = self._norm_q("visual", q)
                    batch["avail"][0, 2], raw_q["visual"] = 1, q
            if not batch["avail"].any():
                return {"available": False}
            batch["visual_mask"][batch["avail"][:, 2] == 0, 0] = True
            tensors = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
            tensors["visual_mask"] = tensors["visual_mask"].bool()
            out = self.fusion(tensors)
        return self._postprocess(out, batch["avail"][0], raw_q)

    def _postprocess(self, out: dict, avail: np.ndarray, raw_q: dict) -> dict[str, Any]:
        logits = out["logits"][0].float().cpu().numpy()
        z = logits / self.temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        branch = out["branch_logits"][0].float().cpu().numpy()
        branch_p = np.exp(branch - branch.max(1, keepdims=True))
        branch_p /= branch_p.sum(1, keepdims=True)
        rel = out["reliability"][0].float().cpu().numpy()
        weights = out["weights"][0].float().cpu().numpy()
        conf = float(probs.max())
        th = self.thresholds
        abstained = conf < th["abstain_confidence"] or float(rel.max()) < th["abstain_reliability"]
        emotion = "uncertain" if abstained else LABELS[int(probs.argmax())]
        valence = float(sum(p * VA_ANCHORS[e][0] for p, e in zip(probs, LABELS, strict=True)))
        arousal = float(sum(p * VA_ANCHORS[e][1] for p, e in zip(probs, LABELS, strict=True)))

        modalities: dict[str, Any] = {}
        for k, m in enumerate(MODALITIES):
            modalities[m] = {
                "available": bool(avail[k]),
                "reliability": round(float(rel[k]), 4),
                "weight": round(float(weights[k]), 4),
                "emotion": LABELS[int(branch_p[k].argmax())] if avail[k] else None,
                "confidence": round(float(branch_p[k].max()), 4) if avail[k] else None,
            }
        conflict = self._conflict(branch_p, rel, avail)
        return {
            "available": True,
            "emotion": emotion,
            "top_emotion": LABELS[int(probs.argmax())],
            "confidence": round(conf, 4),
            "abstained": bool(abstained),
            "probabilities": {e: round(float(p), 4) for e, p in zip(LABELS, probs, strict=True)},
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "conflict": conflict,
            "modalities": modalities,
            "explanation": self._explain(modalities, raw_q, conflict, abstained),
            "model_version": self.version,
        }

    def _conflict(self, branch_p: np.ndarray, rel: np.ndarray, avail: np.ndarray) -> bool:
        th = self.thresholds
        idx = [k for k in range(3) if avail[k] and rel[k] >= th["conflict_reliability"]]
        for i in idx:
            for j in idx:
                if i < j and _js(branch_p[i], branch_p[j]) >= th["conflict_js"]:
                    return True
        return False

    @staticmethod
    def _explain(modalities: dict, raw_q: dict, conflict: bool, abstained: bool) -> list[str]:
        notes: list[str] = []
        names = {"audio": "Voice", "text": "Words", "visual": "Face"}
        for m, info in modalities.items():
            if not info["available"]:
                notes.append(f"{names[m]} unavailable for this segment")
        if "audio" in raw_q and raw_q["audio"][0] < 12:
            notes.append(
                f"Voice down-weighted: background noise (SNR ≈ {raw_q['audio'][0]:.0f} dB)"
            )
        if "text" in raw_q and raw_q["text"][4] > 0:
            notes.append("No words recognized in this segment")
        elif "text" in raw_q and raw_q["text"][0] < -0.8:
            notes.append("Transcript uncertain (low speech-recognition confidence)")
        if "visual" in raw_q:
            q = raw_q["visual"]
            if q[0] < 0.5:
                notes.append("Face visible in only part of the segment")
            if q[3] < 3.0:
                notes.append("Face image is blurry")
            if q[4] < 0.2:
                notes.append("Face is poorly lit")
        avail = {m: i for m, i in modalities.items() if i["available"]}
        if avail:
            top = max(avail, key=lambda m: avail[m]["weight"])
            notes.append(
                f"{names[top]} carried the most weight ({avail[top]['weight'] * 100:.0f}%)"
            )
            emotions = {i["emotion"] for i in avail.values() if i["reliability"] >= 0.5}
            if len(avail) > 1 and len(emotions) == 1:
                notes.append(f"Reliable cues agree on {emotions.pop()}")
        if conflict:
            notes.append("Reliable cues disagree — possible mixed emotion or sarcasm")
        if abstained:
            notes.append("Confidence too low — returned 'uncertain' instead of guessing")
        return notes


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
