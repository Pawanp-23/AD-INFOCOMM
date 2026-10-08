# backend/

FastAPI service for RAMEI: REST + WebSocket API, inference pipeline orchestration and Celery
workers. Spec: [TRD §7](../docs/02-TRD.md#7-backend-specification).

```
backend/
├── app/
│   ├── main.py              # app factory
│   ├── api/v1/
│   │   ├── router.py        # aggregates route modules
│   │   └── routes/          # health, auth, sessions, stream (WS), predict, admin
│   ├── core/                # config, security, logging, exceptions
│   ├── schemas/             # Pydantic request/response models
│   ├── services/            # business logic (sessions, uploads, alerts, summaries, exports)
│   ├── pipeline/            # ingest, VAD, orchestrator, postprocess (calibrate/abstain/smooth/explain)
│   ├── inference/           # model registry, branch runtimes, reliability, fusion (ONNX Runtime)
│   ├── workers/             # Celery app + tasks for offline analysis
│   └── db/
│       ├── models/          # SQLAlchemy ORM models
│       └── repositories/    # data-access layer
└── tests/
    ├── unit/
    └── integration/
```

## Layering rules

`api → services → (pipeline | db.repositories)`, and `pipeline → inference`.
Routes contain no business logic; services never import FastAPI; `inference` never touches the database.
The backend **never trains models** — it only loads versioned bundles exported from [`ml/`](../ml/).

## Run locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

## Test & lint

```bash
pytest
ruff check . && ruff format --check .
mypy app
```
