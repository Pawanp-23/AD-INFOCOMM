# infra/

Deployment assets. Spec: [SAD §9](../docs/03-System-Architecture.md#9-deployment-architecture);
simplifications versus the TRD are recorded in [ADR-0008](../docs/adr/0008-local-first-implementation-choices.md).

```
infra/
├── docker/
│   ├── backend.Dockerfile       # FastAPI + inference engine (CPU torch), non-root, healthcheck
│   └── frontend.Dockerfile      # Vite build -> nginx static site
└── nginx/
    ├── default.conf             # one origin: SPA + /api proxy + WebSocket upgrade
    └── security-headers.conf    # CSP etc., included per location (nginx does not inherit add_header)
```

The root [`docker-compose.yml`](../docker-compose.yml) wires them together:

| Service | Image | Notes |
|---|---|---|
| `web` | `frontend.Dockerfile` (~75 MB) | Public entry point on `:8080` |
| `api` | `backend.Dockerfile` (~4 GB) | Mounts `ml/artifacts/bundles` read-only; SQLite, credentials and model cache in named volumes |
| `postgres` | `postgres:16-alpine` | Optional, `--profile postgres` |

Verified with Docker Desktop 4.95 (Windows 11, WSL2): upload analysis end to end through nginx,
live-stream WebSocket, auth rejection, non-root user, data persisting across restarts.

```bash
docker compose up -d --build
docker compose exec api python -m app.cli create-user --email admin@ramei.local --role admin
docker compose exec api cat /srv/ramei/data/.credentials   # generated password
docker compose logs -f api
docker compose down          # keeps volumes; add -v to delete database and model cache
```

Container-specific settings (already in the image): `RAMEI_ML_ROOT` and `RAMEI_CREDENTIALS_FILE`
point into the writable `/srv/ramei/data` volume; `HOME` and `NUMBA_CACHE_DIR` give libraries a
writable cache. The first analysis downloads the encoder models (~2 GB) into the `hfcache` volume.

Production checklist: real `JWT_SECRET`, TLS in front of nginx, PostgreSQL, backups of the
database volume, `STORE_RAW_MEDIA=false` (default), and a GPU host plus
`INFERENCE_DEVICE=cuda` (CUDA torch wheel) if live latency targets (NFR-01) matter.
