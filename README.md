# RAMEI — Reliability-Aware Adaptive Multimodal Emotion Intelligence

RAMEI recognizes human emotion from **voice, words and face** together, estimates **how
reliable each signal is** for every moment, and **keeps working when signals are noisy,
degraded or missing**. Every prediction comes with a calibrated confidence, an "uncertain"
option instead of guessing, and an explanation of why.

> AD-INFOCOMM internship project. Evolves a single-modal (audio-only) speech emotion
> recognition baseline into a multimodal system with a web platform.

## Documentation

| # | Document | What it answers |
|---|---|---|
| 1 | [Product Requirements (PRD)](docs/01-PRD.md) | What we build, for whom, and how success is measured |
| 2 | [Technical Requirements (TRD)](docs/02-TRD.md) | Models, data, APIs, database, performance, security, testing |
| 3 | [System Architecture](docs/03-System-Architecture.md) | How the components fit together, runtime flows, deployment |
| — | [Architecture Decision Records](docs/adr/) | Why key decisions were made (incl. [ADR-0008](docs/adr/0008-local-first-implementation-choices.md) on implementation choices) |
| — | [Milestone reports](docs/reports/) | Results of each phase (M0 baseline, M1 branches, M2 fusion) |
| — | [Final report](docs/reports/Final-Report.md) | The whole project in one document (Word version built by `scripts/report`) |
| — | [Demo runbook](docs/demo/Demo-Script.md) | Demo preparation, 6-minute video script, likely questions |

## Architecture at a glance

```
 Browser (React + Vite) ──HTTPS / WebSocket──► FastAPI
                                                 ├─ auth, sessions, uploads, live stream
                                                 ├─ Inference engine (versioned model bundle)
                                                 │    speech segments → [WavLM | Whisper+DeBERTa | Face+ViT]
                                                 │    → reliability per modality → reliability-gated fusion
                                                 │    → calibration / abstention → explanation
                                                 └─ SQLite (dev) / PostgreSQL (prod)
```

## Repository structure

| Folder | Contents |
|---|---|
| [`docs/`](docs/) | PRD, TRD, architecture, ADRs, milestone and final reports, demo runbook |
| [`frontend/`](frontend/) | React + Vite + TypeScript dashboard |
| [`backend/`](backend/) | FastAPI service, inference engine, analysis pipeline |
| [`database/`](database/) | Reference schema |
| [`ml/`](ml/) | Data pipelines, models, training, fusion, evaluation, bundle export |
| [`infra/`](infra/) | Dockerfiles and nginx config; [`docker-compose.yml`](docker-compose.yml) runs the stack |
| [`scripts/`](scripts/) | API load test, demo-clip generator, Word report builder |

## Run locally (Windows / macOS / Linux, no Docker needed)

Prerequisites: Python 3.11+, Node 20+, an NVIDIA GPU is recommended (CPU works, slower).

```bash
# 1. ML + backend environment (one virtualenv for both)
cd ml
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -e ".[dev,audio,multimodal]"
pip install -e "../backend[dev,inference]"

# 2. Model bundle (trains on CREMA-D; see ml/README.md and docs/reports for details)
python scripts/download_crema_d.py && python scripts/download_crema_d_video.py
python -m ramei_ml.training.text_branch
python scripts/run_m2.py           # features -> cross-validation -> bundle (~6 h on an RTX 3050, resumable)

# 3. Create a user (password is written to backend/.dev-credentials, git-ignored)
cd ../backend
python -m app.cli create-user --email admin@ramei.local --role admin --name "Admin"

# 4. Start the API (http://127.0.0.1:8000/docs)
uvicorn app.main:app --port 8000

# 5. Start the dashboard (http://localhost:5173)
cd ../frontend && npm install && npm run dev
```

## Run with Docker (alternative)

```bash
cp .env.example .env               # set JWT_SECRET to 32+ random characters
docker compose up --build          # dashboard + API on http://localhost:8080
docker compose exec api python -m app.cli create-user --email admin@ramei.local --role admin
docker compose exec api cat /srv/ramei/data/.credentials     # the generated password
```

The model bundle is mounted read-only from `ml/artifacts/bundles`. Inference runs on CPU in the
container; the first analysis downloads the encoder models (~2 GB) into a cache volume.
Verified with Docker Desktop 4.95 on Windows 11. See [infra/](infra/).

## Results

| Model | Data | Macro-F1 | Notes |
|---|---|---|---|
| Thesis baseline (MFCC + MLP) | RAVDESS, 6-fold speaker CV | 0.512 ± 0.076 | [M0 report](docs/reports/M0-baseline-report.md) |
| Audio branch (WavLM + LoRA) | RAVDESS, 6-fold speaker CV | 0.686 ± 0.028 | [M1 report](docs/reports/M1-unimodal-report.md) |
| Text branch (DeBERTa-v3 + LoRA) | MELD test | 0.448 (weighted-F1 0.591) | [M1 report](docs/reports/M1-unimodal-report.md) |
| Audio branch alone (frozen WavLM) | CREMA-D, 5-fold speaker CV | 0.592 ± 0.033 | 6 classes (no surprise in CREMA-D) |
| **RAMEI fusion (voice + words + face)** | CREMA-D, 5-fold speaker CV | **0.769 ± 0.023** | ECE 0.044; ≤ 10% drop under every corruption — [M2 report](docs/reports/M2-fusion-report.md) |

## Conventions

- Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`; PR review before merge into `main`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/).
- Never commit secrets, datasets, model weights or recordings (see `.gitignore`).
