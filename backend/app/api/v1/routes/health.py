from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import DB, require_role
from app.core.config import get_settings
from app.db.models import User
from app.inference.engine import InferenceEngine
from app.services.analysis import running_jobs

router = APIRouter(tags=["system"])


@router.get("/health")
def health(db: DB) -> dict[str, Any]:
    """Liveness + readiness: database reachable, model bundle present."""
    settings = get_settings()
    db.execute(text("SELECT 1"))
    model = InferenceEngine.status()
    return {
        "status": "ok",
        "env": settings.app_env,
        "database": "ok",
        "model": settings.active_model,
        "model_bundle_found": model["bundle_found"],
        "model_loaded": model["loaded"],
    }


@router.get("/models")
def models(_: Annotated[User, Depends(require_role("analyst"))]) -> dict[str, Any]:
    return {**InferenceEngine.status(), "running_jobs": running_jobs()}
