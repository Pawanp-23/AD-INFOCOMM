"""Live streaming analysis over WebSocket (TRD §7.3).

Client -> server
  text   {"type": "start"} | {"type": "stop"}
  binary 0x01 + PCM16 little-endian mono 16 kHz audio chunk
  binary 0x02 + JPEG video frame (<= 640x480, ~6 fps)
Server -> client
  {"type": "status", "state": "ready" | "processing" | "degraded" | "closed"}
  {"type": "segment_result", ...segment}   every WINDOW_S seconds of audio
  {"type": "alert", ...}                   when an alert rule fires
  {"type": "error", "detail": ...}
"""

import asyncio
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.api.deps import owned_session, user_from_token
from app.db.models import AnalysisSession, Segment
from app.db.session import SessionLocal
from app.inference.engine import InferenceEngine, ModelNotAvailableError, SegmentInput
from app.services.analysis import finalize, segment_to_dict
from app.services.insights import compute_alerts

router = APIRouter(tags=["stream"])
log = logging.getLogger("ramei.stream")
SR = 16_000
WINDOW_S = 3.0
MAX_BACKLOG_S = 6.0  # drop video frames first when analysis falls behind


def _analyze_window(
    engine: InferenceEngine, audio: np.ndarray, jpegs: list[bytes]
) -> dict[str, Any]:
    from ramei_ml.features.audio import AudioConfig, preprocess
    from ramei_ml.features.video import faces_from_frames, sample_frame_indices

    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    transcript, logprob, nospeech, comp = (None, -1.0, 0.0, 1.0)
    clean = None
    if rms > 1e-4:  # skip ASR on digital silence
        clean = preprocess(audio, AudioConfig())
        transcript, logprob, nospeech, comp = engine.transcribe_segment(clean)
    crops = fq = None
    if jpegs:
        idx = sample_frame_indices(len(jpegs), 16)
        frames = []
        for i in idx:
            img = cv2.imdecode(np.frombuffer(jpegs[i], np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)
        with engine.detector_lock:
            crops, fq = faces_from_frames(frames, engine.face_detector)
    result = engine.analyze(
        SegmentInput(
            audio=clean,
            transcript=transcript,
            asr_avg_logprob=logprob,
            asr_no_speech_prob=nospeech,
            asr_compression_ratio=comp,
            face_crops=crops,
            face_quality=fq,
        )
    )
    result["transcript"] = transcript or None
    return result


@router.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    token = websocket.query_params.get("token")
    try:
        with SessionLocal() as db:
            user = user_from_token(db, token)
            session = owned_session(db, session_id, user)
            if session.mode != "live" or session.status == "completed":
                raise HTTPException(409, "Session is not an open live session")
            session.status = "processing"
            db.commit()
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "detail": exc.detail})
        await websocket.close(code=4401 if exc.status_code == 401 else 4403)
        return

    try:
        await websocket.send_json({"type": "status", "state": "loading_models"})
        engine = await run_in_threadpool(InferenceEngine.get)
    except ModelNotAvailableError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1011)
        return
    await websocket.send_json({"type": "status", "state": "ready", "model_version": engine.version})

    audio = np.zeros(0, dtype=np.float32)
    frames: list[tuple[float, bytes]] = []  # (audio time when received, jpeg)
    cursor = 0.0  # seconds of audio already analyzed
    idx = 0
    pending: asyncio.Task | None = None
    results: list[dict[str, Any]] = []
    sent_alerts: set[tuple[str, int]] = set()

    async def run_window(start: float, end: float, i: int) -> None:
        chunk = audio[int(start * SR) : int(end * SR)]
        jpegs = [j for t, j in frames if start <= t < end]
        try:
            result = await run_in_threadpool(_analyze_window, engine, chunk, jpegs)
        except Exception as exc:  # noqa: BLE001 - report and keep the stream alive
            log.exception("window analysis failed")
            await websocket.send_json({"type": "error", "detail": f"analysis failed: {exc}"})
            return
        if not result.get("available"):
            return
        with SessionLocal() as db:
            seg = Segment(
                session_id=session_id,
                idx=i,
                start_ms=int(start * 1000),
                end_ms=int(end * 1000),
                transcript=result.get("transcript"),
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
            db.add(seg)
            s = db.get(AnalysisSession, session_id)
            s.duration_ms = int(end * 1000)
            s.model_version = engine.version
            db.commit()
            payload = segment_to_dict(seg)
        results.append(payload)
        await websocket.send_json(
            {"type": "segment_result", **payload, "model_version": engine.version}
        )
        for a in compute_alerts(results):
            key = (a["rule"], a["start_ms"])
            if key not in sent_alerts:
                sent_alerts.add(key)
                await websocket.send_json({"type": "alert", **a})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                if '"stop"' in msg["text"]:
                    break
                continue
            data = msg.get("bytes") or b""
            if not data:
                continue
            kind, body = data[0], data[1:]
            if kind == 0x01:
                pcm = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2")
                audio = np.concatenate([audio, pcm.astype(np.float32) / 32768.0])
            elif kind == 0x02:
                backlog = audio.size / SR - cursor
                if backlog > MAX_BACKLOG_S:
                    continue  # degrade gracefully: drop video before audio
                frames.append((audio.size / SR, body))
            if audio.size / SR - cursor >= WINDOW_S and (pending is None or pending.done()):
                backlog = audio.size / SR - cursor
                if backlog > MAX_BACKLOG_S:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "state": "degraded",
                            "detail": "analysis is behind; skipping ahead",
                        }
                    )
                    cursor = audio.size / SR - WINDOW_S
                start, end = cursor, cursor + WINDOW_S
                cursor = end
                frames = [(t, j) for t, j in frames if t >= start]
                pending = asyncio.create_task(run_window(start, end, idx))
                idx += 1
    except WebSocketDisconnect:
        pass
    finally:
        if pending is not None:
            await pending
        if results:
            await run_in_threadpool(finalize, session_id)
        else:
            with SessionLocal() as db:
                s = db.get(AnalysisSession, session_id)
                s.status = "completed"
                db.commit()
        try:
            await websocket.send_json(
                {"type": "status", "state": "closed", "segments": len(results)}
            )
            await websocket.close()
        except RuntimeError:
            pass
