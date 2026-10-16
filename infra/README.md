# infra/

Deployment assets. Spec: [SAD §9](../docs/03-System-Architecture.md#9-deployment-architecture);
simplifications versus the TRD are recorded in [ADR-0008](../docs/adr/0008-local-first-implementation-choices.md).

```
infra/
├── docker/
│   ├── backend.Dockerfile    # FastAPI + inference engine (CPU torch), non-root, healthcheck
│   └── frontend.Dockerfile   # Vite build -> nginx static site
└── nginx/
    └── default.conf          # one origin: SPA + /api proxy + WebSocket upgrade + security headers
```

The root [`docker-compose.yml`](../docker-compose.yml) wires them together:

| Service | Image | Notes |
|---|---|---|
| `web` | `frontend.Dockerfile` | Public entry point on `:8080` |
| `api` | `backend.Dockerfile` | Mounts `ml/artifacts/bundles` read-only; SQLite in a named volume |
| `postgres` | `postgres:16-alpine` | Optional, `--profile postgres` |

Production checklist: real `JWT_SECRET`, TLS in front of nginx, PostgreSQL, backups of the
database volume, `STORE_RAW_MEDIA=false` (default), and a GPU host plus
`INFERENCE_DEVICE=cuda` (CUDA torch wheel) if live latency targets (NFR-01) matter.
