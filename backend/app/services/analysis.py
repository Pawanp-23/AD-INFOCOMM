"""Offline analysis of uploaded recordings (TRD §7.4 job lifecycle).

pending -> queued -> processing -> completed | failed. Runs in a worker thread; segments are
written as they are produced so the dashboard can show partial results.
"""

import logging
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.db.models import Alert, AnalysisSession, Segment
from app.db.session import SessionLocal
from app.inference.engine import InferenceEngine, SegmentInput
from app.pipeline import media
from app.services.insights import build_summary, compute_alerts

log = logging.getLogger("ramei.analysis")
_jobs: dict[str, threading.Thread] = {}


def segment_to_dict(s: Segment) -> dict[str, Any]:
    return {
        "id": s.id,
        "index": s.idx,
        "start_ms": s.start_ms,
        "end_ms": s.end_ms,
        "transcript": s.transcript,
        "emotion": s.emotion,
        "confidence": s.confidence,
        "abstained": s.abstained,
        "valence": s.valence,
        "arousal": s.arousal,
        "conflict": s.conflict,
        "probabilities": s.probabilities,
        "modalities": s.modalities,
        "explanation": s.explanation,
    }


def _set(session_id: str, **fields: Any) -> None:
    with SessionLocal() as db:
        s = db.get(AnalysisSession, session_id)
        for k, v in fields.items():
            setattr(s, k, v)
        db.commit()


def finalize(session_id: str) -> None:
    """Compute alerts and summary from stored segments and mark the session completed."""
    with SessionLocal() as db:
        session = db.get(AnalysisSession, session_id)
        segs = [segment_to_dict(s) for s in session.segments]
        for a in session.alerts:
            db.delete(a)
        alerts = compute_alerts(segs)
        for a in alerts:
            db.add(Alert(session_id=session_id, **a))
        used = sorted({m for s in segs for m, i in s["modalities"].items() if i["available"]})
        session.modalities_used = used
        session.summary = build_summary(segs, alerts)
        session.status, session.progress = "completed", 100
        session.completed_at = datetime.now(UTC)
        if segs:
            session.duration_ms = max(session.duration_ms or 0, segs[-1]["end_ms"])
        db.commit()


def analyze_file(session_id: str, path: Path) -> None:
    settings = get_settings()
    try:
        _set(session_id, status="processing", progress=1)
        engine = InferenceEngine.get()
        info = media.validate_media(path, settings.max_upload_minutes)
        audio = media.decode_audio(path)
        from ramei_ml.features.audio import AudioConfig, preprocess

        has_audio = audio.size >= media.SR // 2
        words = engine.transcribe_words(audio) if has_audio else []
        windows = media.speech_segments(audio) if has_audio else []
        if not windows and info["has_video"]:
            windows = media.fixed_windows(info["duration_s"])
        _set(
            session_id,
            duration_ms=int(info["duration_s"] * 1000),
            progress=5,
            model_version=engine.version,
        )

        audio_cfg = AudioConfig(max_seconds=30.0)
        for i, w in enumerate(windows):
            chunk = (
                audio[int(w.start_s * media.SR) : int(w.end_s * media.SR)] if has_audio else None
            )
            seg_words = media.words_in(words, w)
            transcript, logprob, nospeech, comp = None, -1.0, 0.0, 1.0
            if has_audio:
                transcript = " ".join(x["word"] for x in seg_words)
                if seg_words:
                    logprob = float(np.mean([x["seg_logprob"] for x in seg_words]))
                    nospeech = float(np.mean([x["no_speech_prob"] for x in seg_words]))
                    comp = float(np.mean([x["compression_ratio"] for x in seg_words]))
            crops = fq = None
            if info["has_video"]:
                from ramei_ml.features.video import extract_faces

                with engine.detector_lock:
                    crops, fq = extract_faces(
                        path, engine.face_detector, start_s=w.start_s, end_s=w.end_s
                    )
            result = engine.analyze(
                SegmentInput(
                    audio=preprocess(chunk, audio_cfg)
                    if chunk is not None and chunk.size
                    else None,
                    transcript=transcript,
                    asr_avg_logprob=logprob,
                    asr_no_speech_prob=nospeech,
                    asr_compression_ratio=comp,
                    face_crops=crops,
                    face_quality=fq,
                )
            )
            if not result.get("available"):
                continue
            with SessionLocal() as db:
                db.add(
                    Segment(
                        session_id=session_id,
                        idx=i,
                        start_ms=int(w.start_s * 1000),
                        end_ms=int(w.end_s * 1000),
                        transcript=transcript or None,
                        emotion=result["emotion"],
                        confidence=result["confidence"],
                        abstained=result["abstained"],
                        valence=result["valence"],
                        arousal=result["arousal"],
                        conflict=result["conflict"],
                        probabilities=result["probabilities"],
                        modalities=result["modalities"],
                        explanation=result["explanation"],
                    )
                )
                s = db.get(AnalysisSession, session_id)
                s.progress = 5 + int(90 * (i + 1) / len(windows))
                db.commit()
        finalize(session_id)
    except Exception as exc:  # noqa: BLE001 - job boundary: record the failure
        log.error("analysis failed for %s: %s", session_id, traceback.format_exc())
        _set(session_id, status="failed", error_message=str(exc)[:500])
    finally:
        if not settings.store_raw_media:
            path.unlink(missing_ok=True)
        _jobs.pop(session_id, None)


def start_analysis(session_id: str, path: Path) -> None:
    _set(session_id, status="queued", progress=0)
    t = threading.Thread(
        target=analyze_file, args=(session_id, path), daemon=True, name=f"analysis-{session_id[:8]}"
    )
    _jobs[session_id] = t
    t.start()


def running_jobs() -> int:
    return len(_jobs)
