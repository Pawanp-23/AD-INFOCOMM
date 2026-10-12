import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.api.v1.routes.sessions import cleanup_uploads
from app.core.config import get_settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    init_db()
    cleanup_uploads()
    if settings.load_models_on_startup:
        from app.inference.engine import InferenceEngine

        threading.Thread(target=InferenceEngine.get, daemon=True).start()
    yield


def _problem(request: Request, status: int, title: str, detail) -> JSONResponse:
    """RFC 9457 problem details (TRD §7.2)."""
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": request.url.path,
        },
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        description="Reliability-Aware Adaptive Multimodal Emotion Intelligence",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        resp = _problem(request, exc.status_code, "Request failed", exc.detail)
        if exc.headers:
            resp.headers.update(exc.headers)
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return _problem(request, 422, "Validation failed", detail)

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
