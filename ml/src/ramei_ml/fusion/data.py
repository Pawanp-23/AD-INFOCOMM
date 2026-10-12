"""In-memory multimodal feature store with corruption sampling and modality dropout."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ramei_ml.data.labels import EMOTION_TO_ID
from ramei_ml.fusion.extract import AUDIO_VARIANTS, STORE, TEXT_VARIANTS, VISUAL_VARIANTS

# Evaluation scenarios: modality -> variant (None = modality missing)
SCENARIOS: dict[str, dict[str, str | None]] = {
    "clean": {"audio": "clean", "text": "asr_clean", "visual": "clean"},
    "audio_noise_10db": {"audio": "snr10", "text": "asr_clean", "visual": "clean"},
    "noisy_room_0db": {"audio": "snr0", "text": "asr_snr0", "visual": "clean"},
    "visual_blur": {"audio": "clean", "text": "asr_clean", "visual": "blur"},
    "visual_dark": {"audio": "clean", "text": "asr_clean", "visual": "dark"},
    "visual_occluded": {"audio": "clean", "text": "asr_clean", "visual": "occlude"},
    "text_word_drop": {"audio": "clean", "text": "drop50", "visual": "clean"},
    "camera_off": {"audio": "clean", "text": "asr_clean", "visual": None},
    "audio_missing": {"audio": None, "text": "asr_clean", "visual": "clean"},
    "noisy_room_camera_off": {"audio": "snr0", "text": "asr_snr0", "visual": None},
    "audio_only": {"audio": "clean", "text": None, "visual": None},
    "visual_only": {"audio": None, "text": None, "visual": "clean"},
}


@dataclass
class TrainSampling:
    p_clean: float = 0.4  # per modality: probability of the clean variant
    p_drop: float = 0.2  # per modality: probability of removing it (at least one is kept)


class FeatureStore:
    def __init__(self) -> None:
        self.clips = pd.read_csv(STORE / "clips.csv")
        self.y = self.clips["emotion"].map(EMOTION_TO_ID).to_numpy()
        load = lambda name: np.load(STORE / f"{name}.npz", allow_pickle=False)  # noqa: E731
        self.audio = {v: load(f"audio_{v}") for v in AUDIO_VARIANTS}
        self.text = {v: load(f"text_{v}") for v in TEXT_VARIANTS}
        self.visual = {v: load(f"visual_{v}") for v in VISUAL_VARIANTS}
        self.audio = {
            v: {"feats": d["feats"], "quality": d["quality"]} for v, d in self.audio.items()
        }
        self.text = {
            v: {"z": d["z"], "logits": d["logits"], "quality": d["quality"]}
            for v, d in self.text.items()
        }
        self.visual = {
            v: {"cls": d["cls"], "mask": d["mask"], "quality": d["quality"]}
            for v, d in self.visual.items()
        }
        self.q_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit_quality_stats(self, idx: np.ndarray) -> None:
        """Normalize quality features with train-split statistics over all variants."""
        for m, store in (("audio", self.audio), ("text", self.text), ("visual", self.visual)):
            q = np.concatenate([s["quality"][idx] for s in store.values()])
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=-10.0)
            self.q_stats[m] = (q.mean(0), q.std(0) + 1e-6)

    def _q(self, m: str, q: np.ndarray) -> np.ndarray:
        mean, std = self.q_stats[m]
        return ((np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=-10.0) - mean) / std).astype(
            np.float32
        )

    def batch(
        self, idx: np.ndarray, variants: dict[str, list[str | None]], device: str
    ) -> dict[str, torch.Tensor]:
        """Assemble a batch where sample i uses variants[m][i] (None = missing) per modality."""
        n = len(idx)
        out = {
            "audio": np.zeros((n, 13, 1536), np.float32),
            "q_audio": np.zeros((n, 5), np.float32),
            "text_z": np.zeros((n, 768), np.float32),
            "text_logits": np.zeros((n, 7), np.float32),
            "q_text": np.zeros((n, 5), np.float32),
            "visual": np.zeros((n, 16, 768), np.float32),
            "visual_mask": np.zeros((n, 16), bool),
            "q_visual": np.zeros((n, 6), np.float32),
            "avail": np.zeros((n, 3), np.float32),
        }
        for j, (i, va, vt, vv) in enumerate(
            zip(idx, variants["audio"], variants["text"], variants["visual"], strict=True)
        ):
            if va is not None:
                out["audio"][j] = self.audio[va]["feats"][i]
                out["q_audio"][j] = self._q("audio", self.audio[va]["quality"][i])
                out["avail"][j, 0] = 1
            if vt is not None:
                out["text_z"][j] = self.text[vt]["z"][i]
                out["text_logits"][j] = self.text[vt]["logits"][i]
                out["q_text"][j] = self._q("text", self.text[vt]["quality"][i])
                out["avail"][j, 1] = 1
            if vv is not None:
                mask = self.visual[vv]["mask"][i]
                if mask.any():
                    out["visual"][j] = self.visual[vv]["cls"][i]
                    out["visual_mask"][j] = mask
                    out["q_visual"][j] = self._q("visual", self.visual[vv]["quality"][i])
                    out["avail"][j, 2] = 1
        out["visual_mask"][out["avail"][:, 2] == 0, 0] = True  # keep attention well-defined
        t = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
        t["visual_mask"] = t["visual_mask"].bool()
        return t

    def scenario_variants(self, n: int, scenario: str) -> dict[str, list[str | None]]:
        s = SCENARIOS[scenario]
        return {m: [s[m]] * n for m in ("audio", "text", "visual")}

    @staticmethod
    def random_variants(
        n: int, rng: np.random.Generator, sampling: TrainSampling
    ) -> dict[str, list[str | None]]:
        options = {
            "audio": list(AUDIO_VARIANTS),
            "text": list(TEXT_VARIANTS),
            "visual": list(VISUAL_VARIANTS),
        }
        out: dict[str, list[str | None]] = {}
        for m, opts in options.items():
            clean, corrupted = opts[0], opts[1:]
            out[m] = [
                clean if rng.random() < sampling.p_clean else str(rng.choice(corrupted))
                for _ in range(n)
            ]
        for i in range(n):  # modality dropout, never all three
            dropped = [m for m in options if rng.random() < sampling.p_drop]
            if len(dropped) == 3:
                dropped = dropped[:2]
            for m in dropped:
                out[m][i] = None
        return out
