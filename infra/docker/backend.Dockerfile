# RAMEI API + inference (CPU). Build from the repository root:
#   docker build -f infra/docker/backend.Dockerfile -t ramei-api .
# The model bundle is mounted at runtime (ml/artifacts/bundles), never baked into the image.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/var/cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ramei
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Dependencies first (cached layer), then sources.
COPY ml/pyproject.toml ml/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
COPY ml/src ml/src
COPY backend/app backend/app
RUN pip install "./ml[multimodal]" "./backend[inference,postgres,llm]"

RUN useradd --system --uid 10001 ramei \
 && mkdir -p /srv/ramei/backend/uploads /srv/ramei/data "$HF_HOME" \
 && chown -R ramei /srv/ramei/backend/uploads /srv/ramei/data "$HF_HOME"
USER ramei

WORKDIR /srv/ramei/backend
ENV APP_ENV=production \
    DATABASE_URL=sqlite:////srv/ramei/data/ramei.db \
    MODEL_BUNDLE_DIR=/srv/ramei/bundles \
    UPLOAD_DIR=/srv/ramei/backend/uploads \
    INFERENCE_DEVICE=cpu

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
# One worker: the inference engine holds the models in memory (ADR-0008).
# The API port is only reachable on the internal network, so trust nginx's X-Forwarded-For
# (the login rate limiter keys on the real client address). Access logging happens in nginx
# without query strings, because the live-stream WebSocket carries its token there.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
