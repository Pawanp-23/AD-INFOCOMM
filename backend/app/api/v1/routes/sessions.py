import csv
import io
import json
import shutil
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import DB, CurrentUser, owned_session
from app.core.config import get_settings
from app.db.models import Alert, AnalysisSession, AuditLog, Segment
from app.pipeline.media import AUDIO_EXTS, VIDEO_EXTS, MediaError, validate_media
from app.schemas import AlertOut, Page, SegmentOut, SessionCreate, SessionDetail, SessionOut
from app.services.analysis import segment_to_dict, start_analysis

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _out(db: DB, s: AnalysisSession, detail: bool = False) -> SessionOut:
    n_seg = db.scalar(select(func.count()).where(Segment.session_id == s.id)) or 0
    n_alert = db.scalar(select(func.count()).where(Alert.session_id == s.id)) or 0
    data = SessionOut.model_validate(s).model_dump()
    data.update(
        segment_count=n_seg,
        alert_count=n_alert,
        dominant_emotion=(s.summary or {}).get("dominant"),
        owner_email=s.user.email if s.user else None,
    )
    if detail:
        return SessionDetail(**data, summary=s.summary)
    return SessionOut(**data)


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(body: SessionCreate, db: DB, user: CurrentUser) -> SessionOut:
    s = AnalysisSession(
        org_id=user.org_id,
        user_id=user.id,
        mode=body.mode,
        title=body.title or None,
        status="pending",
        consent_text_version="1.0",
    )
    db.add(s)
    db.flush()
    db.add(
        AuditLog(
            org_id=user.org_id,
            user_id=user.id,
            action="consent.granted",
            resource="session",
            resource_id=s.id,
            details={"mode": body.mode},
        )
    )
    db.commit()
    return _out(db, s)


@router.get("", response_model=Page)
def list_sessions(
    db: DB,
    user: CurrentUser,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    q: str | None = None,
    limit: Annotated[int, Query(le=100)] = 20,
    offset: int = 0,
) -> Page:
    stmt = select(AnalysisSession).where(AnalysisSession.org_id == user.org_id)
    if user.role == "agent":
        stmt = stmt.where(AnalysisSession.user_id == user.id)
    if status_filter:
        stmt = stmt.where(AnalysisSession.status == status_filter)
    if q:
        stmt = stmt.where(AnalysisSession.title.ilike(f"%{q}%"))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(AnalysisSession.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return Page(items=[_out(db, s) for s in rows], total=total)


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, db: DB, user: CurrentUser) -> SessionDetail:
    return _out(db, owned_session(db, session_id, user), detail=True)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, db: DB, user: CurrentUser) -> Response:
    s = owned_session(db, session_id, user)
    if s.user_id != user.id and user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner or an admin can delete")
    if s.status in ("queued", "processing"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Analysis still running")
    db.add(
        AuditLog(
            org_id=user.org_id,
            user_id=user.id,
            action="session.delete",
            resource="session",
            resource_id=s.id,
        )
    )
    db.delete(s)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/media", response_model=SessionOut, status_code=status.HTTP_202_ACCEPTED)
def upload_media(
    session_id: str, db: DB, user: CurrentUser, file: Annotated[UploadFile, File()]
) -> SessionOut:
    settings = get_settings()
    s = owned_session(db, session_id, user)
    if s.mode != "upload" or s.status not in ("pending", "failed"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Session does not accept uploads")
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in AUDIO_EXTS | VIDEO_EXTS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Supported: {', '.join(sorted(AUDIO_EXTS | VIDEO_EXTS))}",
        )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"  # never trust the client filename
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"File larger than {settings.max_upload_mb} MB",
                )
            out.write(chunk)
    try:
        validate_media(dest, settings.max_upload_minutes)
    except MediaError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if not s.title:
        s.title = (file.filename or "Recording")[:200]
    db.commit()
    start_analysis(s.id, dest)
    db.refresh(s)
    return _out(db, s)


@router.get("/{session_id}/segments", response_model=list[SegmentOut])
def get_segments(session_id: str, db: DB, user: CurrentUser) -> list[dict]:
    s = owned_session(db, session_id, user)
    return [segment_to_dict(x) for x in s.segments]


@router.get("/{session_id}/alerts", response_model=list[AlertOut])
def get_alerts(session_id: str, db: DB, user: CurrentUser) -> list[Alert]:
    return list(owned_session(db, session_id, user).alerts)


@router.post("/{session_id}/alerts/{alert_id}/ack", response_model=AlertOut)
def acknowledge_alert(session_id: str, alert_id: str, db: DB, user: CurrentUser) -> Alert:
    s = owned_session(db, session_id, user)
    alert = next((a for a in s.alerts if a.id == alert_id), None)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
    alert.acknowledged = True
    db.commit()
    return alert


@router.get("/{session_id}/export")
def export_session(
    session_id: str,
    db: DB,
    user: CurrentUser,
    format: Annotated[str, Query(pattern="^(json|csv)$")] = "json",
) -> Response:
    s = owned_session(db, session_id, user)
    segs = [segment_to_dict(x) for x in s.segments]
    db.add(
        AuditLog(
            org_id=user.org_id,
            user_id=user.id,
            action="session.export",
            resource="session",
            resource_id=s.id,
            details={"format": format},
        )
    )
    db.commit()
    name = f"ramei-session-{s.id[:8]}"
    if format == "json":
        body = json.dumps(
            {"session": _out(db, s, detail=True).model_dump(mode="json"), "segments": segs},
            indent=2,
        )
        return Response(
            body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "start_ms",
            "end_ms",
            "emotion",
            "confidence",
            "abstained",
            "valence",
            "arousal",
            "conflict",
            "audio_reliability",
            "text_reliability",
            "visual_reliability",
            "transcript",
        ]
    )
    for x in segs:
        m = x["modalities"]
        w.writerow(
            [
                x["start_ms"],
                x["end_ms"],
                x["emotion"],
                x["confidence"],
                x["abstained"],
                x["valence"],
                x["arousal"],
                x["conflict"],
                m["audio"]["reliability"],
                m["text"]["reliability"],
                m["visual"]["reliability"],
                x["transcript"] or "",
            ]
        )
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )


def cleanup_uploads() -> None:
    """Remove leftover uploads from interrupted runs (privacy default)."""
    settings = get_settings()
    if settings.upload_dir.exists() and not settings.store_raw_media:
        shutil.rmtree(settings.upload_dir, ignore_errors=True)
