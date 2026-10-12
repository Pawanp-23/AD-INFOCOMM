# ADR-0008: Local-first implementation choices for the internship build

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-10-13 |
| Deciders | Project team |

## Context

The TRD describes a production target (PostgreSQL, Redis, Celery, ONNX Runtime, Silero VAD,
Docker). The development machine has an RTX 3050 (4 GB), no Docker, and the build had to be
delivered quickly. Several TRD items were implemented in a simpler form that keeps the same
contracts, so the production path stays open.

## Decisions

| TRD item | Implemented as | Why | Path to the TRD target |
|---|---|---|---|
| PostgreSQL 16 | SQLite via the same SQLAlchemy models (`DATABASE_URL` switches) | No Docker / Postgres on the dev machine | Set `DATABASE_URL=postgresql+psycopg://…`; schema in `database/schema/schema.sql` |
| Celery + Redis workers | In-process worker threads with persisted job status | One machine, one GPU; avoids two extra services | Move `analyze_file` into a Celery task; API code unchanged |
| ONNX Runtime serving | PyTorch inference from a versioned bundle | WavLM / ViT / DeBERTa export + validation would delay delivery | Export encoders + fusion to ONNX; engine interface unchanged |
| Silero VAD | Energy-based segmentation (librosa) into 1–6 s windows | No extra model download; adequate for single-speaker recordings | Swap `pipeline/media.speech_segments` |
| Fusion data: RAVDESS A/V | CREMA-D audio + video (91 actors) | Zenodo throttled RAVDESS video to ~8 KB/s; CREMA-D gives 4× more speakers | RAVDESS downloader kept (`scripts/download_ravdess_video.py`) |
| Fine-tuned encoders inside fusion | Frozen WavLM / ViT, trained heads; text encoder fine-tuned on MELD | Fast speaker-independent CV on cached features; frozen WavLM matched LoRA in E2a | Fine-tune end-to-end once more GPU memory is available |
| Valence/arousal regression head | Probability-weighted circumplex estimate | CREMA-D / MELD have no V/A labels | Train the head on IEMOCAP / MSP-Podcast once licensed |
| LLM session summary | Deterministic data-driven summary | No approved LLM provider yet (PRD Q2) | Plug a provider into `services/insights.build_summary` |
| Refresh token in httpOnly cookie | Access token in `sessionStorage` (8 h) | Single-host dev deployment | Add `/auth/refresh` + cookie |

## Consequences

- Every component still honors its TRD interface (REST/WS contracts, bundle manifest, schema).
- Production hardening items are tracked in the M4 checklist of the PRD release plan.
