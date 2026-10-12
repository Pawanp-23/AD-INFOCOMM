from collections import Counter

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import DB, CurrentUser
from app.db.models import Alert, AnalysisSession, Segment
from app.schemas import DashboardStats

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def stats(db: DB, user: CurrentUser) -> DashboardStats:
    sessions = select(AnalysisSession).where(AnalysisSession.org_id == user.org_id)
    if user.role == "agent":
        sessions = sessions.where(AnalysisSession.user_id == user.id)
    rows = db.scalars(sessions).all()
    ids = [s.id for s in rows]
    segs = db.scalars(select(Segment).where(Segment.session_id.in_(ids))).all() if ids else []
    alerts = (
        db.scalars(
            select(Alert).where(Alert.session_id.in_(ids)).order_by(Alert.created_at.desc())
        ).all()
        if ids
        else []
    )
    total_ms = sum(s.end_ms - s.start_ms for s in segs)
    dist = Counter()
    for s in segs:
        dist[s.emotion] += s.end_ms - s.start_ms
    reliab = {}
    for m in ("audio", "text", "visual"):
        vals = [s.modalities[m]["reliability"] for s in segs if s.modalities[m]["available"]]
        reliab[m] = round(sum(vals) / len(vals), 3) if vals else 0.0
    titles = {s.id: s.title for s in rows}
    return DashboardStats(
        sessions_total=len(rows),
        sessions_completed=sum(s.status == "completed" for s in rows),
        hours_analyzed=round(total_ms / 3_600_000, 2),
        alerts_total=len(alerts),
        critical_alerts=sum(a.severity == "critical" for a in alerts),
        emotion_distribution={e: round(v / max(total_ms, 1), 3) for e, v in dist.items()},
        mean_reliability=reliab,
        recent_alerts=[
            {
                "id": a.id,
                "session_id": a.session_id,
                "session_title": titles.get(a.session_id),
                "rule": a.rule,
                "severity": a.severity,
                "message": a.message,
                "start_ms": a.start_ms,
                "created_at": a.created_at.isoformat(),
            }
            for a in alerts[:8]
        ],
    )
