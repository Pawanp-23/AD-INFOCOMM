from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Readiness checks (DB, Redis, model bundle) are added with those modules."""
    settings = get_settings()
    return {"status": "ok", "env": settings.app_env, "model": settings.active_model}
