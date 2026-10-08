# RAMEI — Reliability-Aware Adaptive Multimodal Emotion Intelligence

RAMEI recognizes human emotion from **voice, words and face** together, estimates **how
reliable each signal is** for every moment, and **adaptively shifts trust** toward the
reliable ones. Every prediction comes with a calibrated confidence, an "uncertain" option
instead of guessing, and an explanation of why.

> AD-INFOCOMM internship project. Evolves the single-modal (audio-only) speech emotion
> recognition baseline into a production-grade multimodal system.

## Documentation

Read in order:

| # | Document | What it answers |
|---|---|---|
| 1 | [Product Requirements (PRD)](docs/01-PRD.md) | What we build, for whom, and how success is measured |
| 2 | [Technical Requirements (TRD)](docs/02-TRD.md) | Models, data, APIs, database, performance, security, testing |
| 3 | [System Architecture](docs/03-System-Architecture.md) | How the components fit together, runtime flows, deployment |
| — | [Architecture Decision Records](docs/adr/) | Why key decisions were made |

## Architecture at a glance

```
 Browser (React)   ──HTTPS/WSS──► FastAPI ──► Pipeline: VAD → [Audio | ASR+Text | Face] branches
                                     │                         → Reliability module → Adaptive fusion
                                     │                         → Calibrate / abstain → Explain
                                     ├──► PostgreSQL (sessions, segments, alerts)
                                     ├──► Redis (Celery broker, live state)
                                     └──► Model bundles (exported from ml/)
```

## Repository structure

| Folder | Contents |
|---|---|
| [`docs/`](docs/) | PRD, TRD, architecture, ADRs, diagrams |
| [`frontend/`](frontend/) | React + Vite + TypeScript dashboard |
| [`backend/`](backend/) | FastAPI service, inference pipeline, Celery workers |
| [`database/`](database/) | Schema, migrations, seeds |
| [`ml/`](ml/) | Datasets, model training, evaluation, ONNX export |
| [`infra/`](infra/) | Dockerfiles, reverse proxy, monitoring |
| [`scripts/`](scripts/) | Developer utilities |

## Quick start (infrastructure)

```bash
cp .env.example .env
docker compose up -d
```

This starts PostgreSQL (schema + dev seed applied), Redis and MinIO. Application services
are added to `docker-compose.yml` as they are implemented (see milestones in the PRD §10).

## Tech stack

PyTorch · Hugging Face (WavLM, Whisper, DeBERTa, ViT) · ONNX Runtime · FastAPI · Celery ·
PostgreSQL · Redis · React · Vite · TypeScript · Tailwind · Docker · GitHub Actions · MLflow

## Conventions

- Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`; PR review before merge into `main`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/).
- Never commit secrets, datasets, model weights or recordings (see `.gitignore`).
