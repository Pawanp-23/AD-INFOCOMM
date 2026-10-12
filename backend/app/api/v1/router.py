from fastapi import APIRouter

from app.api.v1.routes import auth, dashboard, health, sessions, stream

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(sessions.router)
api_router.include_router(dashboard.router)
api_router.include_router(stream.router)
