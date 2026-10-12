"""Speech-to-text for the text branch (TRD §4.2) with faster-whisper.

Besides the transcript, Whisper's own confidence signals are kept: they are the text
modality's quality features for the reliability module (TRD §4.4).

CLI (from ml/):
    python -m ramei_ml.features.asr --manifest data/manifests/ravdess_audio.csv \
        --out data/processed/transcripts_ravdess.csv
"""

import argparse
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ramei_ml.features.audio import AudioConfig, load_audio
from ramei_ml.paths import resolve


@dataclass
class Transcript:
    text: str
    avg_logprob: float  # mean token log-probability (higher = more confident)
    no_speech_prob: float  # probability the segment contains no speech
    compression_ratio: float  # high values indicate repetitive / hallucinated output
    n_words: int


class Transcriber:
    def __init__(
        self, model: str = "small.en", device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """audio: mono float32 at 16 kHz."""
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        segments = list(segments)
        if not segments:
            return Transcript("", -math.inf, 1.0, 0.0, 0)
        text = " ".join(s.text.strip() for s in segments).strip()
        weights = np.array([max(s.end - s.start, 1e-3) for s in segments])
        avg = lambda xs: float(np.average(xs, weights=weights))  # noqa: E731
        return Transcript(
            text=text,
            avg_logprob=avg([s.avg_logprob for s in segments]),
            no_speech_prob=avg([s.no_speech_prob for s in segments]),
            compression_ratio=avg([s.compression_ratio for s in segments]),
            n_words=len(text.split()),
        )


def transcribe_manifest(
    manifest: pd.DataFrame, transcriber: Transcriber, audio_cfg: AudioConfig = AudioConfig()
) -> pd.DataFrame:
    rows = []
    for i, path in enumerate(manifest["path"]):
        t = transcriber.transcribe(load_audio(path, audio_cfg))
        rows.append({"path": path, **asdict(t)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(manifest)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    manifest = pd.read_csv(resolve(args.manifest))
    df = transcribe_manifest(manifest, Transcriber(args.model, args.device, args.compute_type))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} transcripts -> {out}")
    print(df["text"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
