# backend/

FastAPI service for RAMEI: authentication, sessions, uploads, live streaming and the inference
engine. Spec: [TRD §7](../docs/02-TRD.md#7-backend-specification) ·
Implementation choices: [ADR-0008](../docs/adr/0008-local-first-implementation-choices.md).

```
backend/
├── app/
│   ├── main.py                 # app factory, CORS, problem-details errors, startup
│   ├── cli.py                  # python -m app.cli create-user ...
│   ├── api/
│   │   ├── deps.py             # DB session, current user, role checks, org/owner scoping
│   │   └── v1/routes/          # health, auth, sessions, dashboard, stream (WebSocket)
│   ├── core/                   # settings (env), security (Argon2id, JWT)
│   ├── db/                     # SQLAlchemy engine + models (mirror database/schema/schema.sql)
│   ├── schemas/                # Pydantic request/response models
│   ├── inference/engine.py     # loads the model bundle; segment -> fused, calibrated, explained result
│   ├── pipeline/media.py       # ffmpeg decoding, validation, speech segmentation
│   └── services/               # analysis jobs, alerts + summary, rate limiting
└── tests/                      # unit + API integration tests (no models needed)
```

## Endpoints (prefix `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + readiness (DB, model bundle) |
| POST | `/auth/login` | Email + password → JWT |
| GET | `/auth/me` | Current user |
| POST/GET | `/sessions` | Create (consent required) / list (search, status filter, paging) |
| GET/DELETE | `/sessions/{id}` | Detail with summary / delete |
| POST | `/sessions/{id}/media` | Upload audio/video → background analysis |
| GET | `/sessions/{id}/segments` · `/alerts` · `/export?format=json\|csv` | Results |
| POST | `/sessions/{id}/alerts/{alert_id}/ack` | Acknowledge an alert |
| GET | `/dashboard/stats` | Overview KPIs |
| GET | `/models` | Active model bundle + manifest (analyst+) |
| WS | `/stream/{id}?token=…` | Live analysis (PCM16 audio `0x01` + JPEG frames `0x02`) |

Interactive docs: http://127.0.0.1:8000/docs

## Run

Uses the ML virtualenv (it provides torch and the `ramei_ml` package):

```bash
cd backend
../ml/.venv/Scripts/python -m pip install -e ".[dev,inference]"
../ml/.venv/Scripts/python -m app.cli create-user --email admin@ramei.local --role admin
../ml/.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

| Setting (env) | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/ramei.db` | PostgreSQL in production |
| `JWT_SECRET` | dev value | **Required** outside `APP_ENV=local` |
| `MODEL_BUNDLE_DIR` / `ACTIVE_MODEL` | `ml/artifacts/bundles` / `ramei-rgcmt-1.0.0` | Built by `ml/scripts/run_m2.py` |
| `INFERENCE_DEVICE` | `auto` | `cuda` if available |
| `STORE_RAW_MEDIA` | `false` | Uploads are deleted after analysis |

## Test & lint

```bash
../ml/.venv/Scripts/python -m pytest
../ml/.venv/Scripts/python -m ruff check .
```
