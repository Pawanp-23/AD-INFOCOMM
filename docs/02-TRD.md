# Technical Requirements Document (TRD)

## RAMEI — Reliability-Aware Adaptive Multimodal Emotion Intelligence

| Field | Value |
|---|---|
| Document ID | RAMEI-TRD-001 |
| Version | 1.0 |
| Status | Draft for review |
| Date | 2026-10-09 |
| Owner | Technical Lead — AD-INFOCOMM internship team |
| Implements | [01-PRD.md](01-PRD.md) |
| Architecture | [03-System-Architecture.md](03-System-Architecture.md) |

---

## 1. Purpose and scope

This document translates the PRD into precise technical requirements: technology choices,
data specifications, model specifications, API contracts, database schema, performance
budgets, security controls, testing and delivery practices. Every requirement here traces
back to a PRD requirement (see §17).

## 2. Technology stack

| Layer | Technology | Version (min) | Rationale |
|---|---|---|---|
| Language (ML/backend) | Python | 3.11 | Ecosystem, typing, performance |
| Deep learning | PyTorch | 2.3 | Industry standard, ONNX export |
| Pretrained models | Hugging Face Transformers, PEFT | 4.44 / 0.12 | WavLM, Whisper, DeBERTa, ViT, LoRA |
| Audio | torchaudio, librosa, Silero VAD | — | Loading, features, segmentation |
| ASR | faster-whisper (CTranslate2) | 1.0 | 4× faster Whisper inference |
| Vision | OpenCV, MediaPipe Face Detection | 4.9 / 0.10 | Face detection & tracking |
| Experiment tracking | MLflow | 2.14 | Runs, metrics, model registry |
| Config | Hydra | 1.3 | Composable experiment configs |
| Inference runtime | ONNX Runtime (CPU/CUDA) | 1.18 | Low-latency serving |
| API | FastAPI + Uvicorn | 0.111 | Async, OpenAPI, WebSockets |
| Validation | Pydantic | 2.x | Typed request/response schemas |
| Task queue | Celery + Redis | 5.4 / 7 | Async processing of uploaded files |
| Database | PostgreSQL | 16 | Relational integrity, JSONB |
| ORM / migrations | SQLAlchemy 2 + Alembic | 2.0 / 1.13 | Typed ORM, versioned schema |
| Cache / pub-sub | Redis | 7 | Stream state, rate limits, broker |
| Object storage | MinIO (S3-compatible) | latest stable | Optional raw media storage |
| Frontend | React + Vite + TypeScript | 18 / 5 / 5 | Fast dev server and builds; an authenticated dashboard needs no SSR/SEO; deploys as static files |
| Routing | React Router | 6 | Client-side routing with lazy-loaded routes and auth guards |
| UI | Tailwind CSS, shadcn/ui, Recharts | 3 / — / 2 | Accessible components, charts |
| Client state | TanStack Query, Zustand | 5 / 4 | Server cache, live state |
| Containers | Docker, Docker Compose | 24 / 2 | One-command local stack |
| CI/CD | GitHub Actions | — | Lint, test, build on every PR |
| Observability | Prometheus, Grafana, structured JSON logs | — | Metrics, dashboards |

## 3. Data requirements

### 3.1 Datasets

| Dataset | Modalities | Size | Labels | Use | License |
|---|---|---|---|---|---|
| RAVDESS | Audio + Video | 24 actors, ~7.3k files | 8 emotions | Audio, visual, A/V fusion | CC BY-NC-SA 4.0 |
| CREMA-D | Audio + Video | 91 actors, 7.4k clips | 6 emotions | Audio, visual | ODbL |
| TESS | Audio | 2.8k | 7 emotions | Audio pretraining | CC BY-NC-ND |
| IEMOCAP | Audio + Video + Text | ~12 h, 10 speakers | Categorical + V/A/D | Main fusion benchmark | Academic license (request) |
| MELD | Audio + Video + Text | 13k utterances | 7 emotions | Natural conversational fusion | GPL-3.0 |
| CMU-MOSEI | Audio + Video + Text | 23k segments | 6 emotions + sentiment | Fusion, robustness | Research use |
| GoEmotions | Text | 58k | 27 → mapped to 7 | Text branch | Apache 2.0 |
| AffectNet / FER+ | Image | 400k / 35k | 8 / 8 | Visual branch | Academic license / MIT |

### 3.2 Unified label space

All datasets are mapped to **7 classes**:
`neutral, happy, sad, angry, fear, disgust, surprise`.
`calm → neutral`, `contempt → disgust`, `excited → happy`, `frustrated → angry`.
Unmappable labels are excluded. Valence/arousal are used where available (IEMOCAP, MSP-style
annotations) and normalized to [-1, 1].

### 3.3 Splits (mandatory)

- **Speaker-independent splits** for every dataset: no speaker appears in more than one of train/val/test.
- IEMOCAP: leave-one-session-out (5-fold).
- MELD: official train/dev/test.
- **Cross-corpus test:** train on {RAVDESS, CREMA-D, MELD}, test on held-out IEMOCAP.
- Split manifests are versioned in `ml/data/manifests/` (CSV with `path, speaker_id, split, label`).

### 3.4 Preprocessing

| Modality | Steps |
|---|---|
| Audio | Mono, resample 16 kHz, loudness normalize (−23 LUFS), Silero VAD, segments 1–8 s (pad/crop to max 8 s) |
| Text | faster-whisper transcript (word timestamps), lowercase-preserving, max 128 tokens |
| Video | Sample 6 fps, MediaPipe face detect, crop + align 224×224, keep up to 16 frames per segment |
| Alignment | Each segment = VAD utterance window; text and frames aligned by timestamps |

### 3.5 Corruption augmentation (for reliability & adaptivity)

| Modality | Corruptions (random severity 0–4) |
|---|---|
| Audio | Additive noise (MUSAN/ESC-50) at SNR 20→0 dB, reverb (RIR), clipping, band-limiting to 8 kHz |
| Text | Word drop/substitution simulating ASR WER 0→40% |
| Video | Gaussian blur, low light, occlusion patches, JPEG artifacts, frame drop |
| All | **Modality dropout** p = 0.3 (whole modality removed) |

## 4. Model specifications

### 4.1 Audio branch

| Item | Specification |
|---|---|
| Input | 16 kHz waveform, ≤ 8 s |
| Encoder | `microsoft/wavlm-base-plus` (alt: `emotion2vec_base`) |
| Adaptation | LoRA (r=8, α=16) on attention projections; layer-weighted sum of hidden states |
| Pooling | Attentive statistics pooling → 768-d `z_a` |
| Head | Linear(768→256) → GELU → Dropout(0.2) → Linear(256→7) |
| Output | `z_a ∈ R^768`, logits `ℓ_a ∈ R^7` |

### 4.2 Text branch

| Item | Specification |
|---|---|
| ASR | faster-whisper `small.en` (live) / `medium.en` (offline), returns avg log-prob, no-speech prob |
| Encoder | `microsoft/deberta-v3-base` fine-tuned (alt: `roberta-base`) |
| Pooling | [CLS] → 768-d `z_t` |
| Head | Linear(768→7) |
| Output | `z_t`, `ℓ_t`, ASR confidence features |

### 4.3 Visual branch

| Item | Specification |
|---|---|
| Input | Up to 16 aligned face crops 224×224 |
| Frame encoder | ViT-B/16 pretrained on AffectNet/FER+ (alt: ResNet-50) |
| Temporal | 2-layer Transformer encoder over frame embeddings + [CLS] |
| Output | `z_v ∈ R^768`, `ℓ_v ∈ R^7`, face quality features |

### 4.4 Reliability Estimation Module (REM)

For each modality *m* ∈ {a, t, v}:

**Quality features q_m**

| Modality | Features |
|---|---|
| Audio | estimated SNR (dB), speech ratio from VAD, clipping ratio, RMS loudness, duration |
| Text | Whisper avg log-prob, no-speech prob, word count, compression ratio |
| Visual | mean face-detection confidence, face area ratio, Laplacian blur variance, mean brightness, |yaw|/|pitch|, face-present ratio |

**Uncertainty features u_m** (from the branch prediction)

- Predictive entropy `H(softmax(ℓ_m))`
- Max-probability and top-2 margin
- MC-dropout variance over K=5 passes (offline only; live uses entropy + margin)

**Model**

```
r_m = σ( MLP_m( [ normalize(q_m) ; u_m ; proj(z_m) ] ) )      r_m ∈ [0, 1]
MLP_m: Linear(d→64) → ReLU → Linear(64→1)
```

**Training target:** `y_m = 1` if the unimodal prediction `argmax ℓ_m` equals the ground truth, else 0.
Loss: binary cross-entropy, trained on corrupted + clean data with encoders frozen.

**Missing modality:** `r_m = 0`, `z_m = 0`, and an availability mask `a_m = 0` are passed to fusion.

### 4.5 Adaptive fusion

Three implementations (selected via config); the reliability-gated transformer is the primary model.

| Variant | Formula / design |
|---|---|
| F1 — Reliability-weighted late fusion | `w_m = a_m·r_m / Σ a_k·r_k`; `p = Σ w_m · softmax(ℓ_m / T_m)` |
| F2 — Gated embedding fusion | `w = softmax(W_g [z_a; z_t; z_v; r_a; r_t; r_v] − ∞·(1−a))`; `z = Σ w_m · P_m z_m` |
| **F3 — Reliability-Gated Cross-Modal Transformer (RG-CMT)** | Tokens `x_m = P_m z_m + e_m` (modality embedding). Attention logits get an additive bias `β · log(r_k + ε)` for key token *k*, and `−∞` for missing modalities. 2 layers, 4 heads, d=256, learnable [CLS]. |

**Output heads (on fused representation)**

- Emotion: Linear(256→7), cross-entropy with label smoothing 0.1 and class weights.
- Valence/Arousal: Linear(256→2) + tanh, loss `1 − CCC`.
- Total loss: `L = L_ce + λ_va · L_ccc + λ_aux · Σ_m L_ce(ℓ_m)` with λ_va = 0.5, λ_aux = 0.3.

**Reported weights:** for explainability, the effective modality weights are the [CLS] attention
mass per modality averaged over heads and layers (F3) or `w` (F1/F2).

### 4.6 Calibration and abstention

- **Temperature scaling** fitted on the validation set (one scalar T for the fused head).
- **Confidence** `c = max softmax(ℓ_fused / T)`.
- **Abstain** when `c < τ_c` (default 0.45) **or** `max_m r_m < τ_r` (default 0.30). Output label `"uncertain"`.
- **Conflict flag** when two modalities with `r_m ≥ 0.6` have Jensen–Shannon divergence ≥ 0.5.
- Thresholds are tuned on validation to reach ≥ 80% coverage with maximum selective accuracy.

### 4.7 Temporal smoothing

- Live: exponential moving average over fused probabilities, `α = 0.6`.
- Offline: 1-layer bidirectional GRU over the segment sequence (optional, config-controlled).

### 4.8 Explainability

| Output | Method |
|---|---|
| Modality weights | From fusion (§4.5) |
| Reliability reasons | Rule templates over quality features (e.g. `SNR < 5 dB → "audio is noisy"`) |
| Audio saliency | Integrated Gradients over waveform frames → top time regions |
| Text attribution | Integrated Gradients over tokens → top-k words |
| Visual saliency | Grad-CAM on the last ViT block → heatmap of key frame |

### 4.9 Session intelligence

- **Alerts:** rules over the smoothed timeline (e.g. `angry ≥ 20 s` or `valence < −0.5 for ≥ 30 s`), configurable per organization.
- **Summary:** transcript + segment table + alerts sent to a configured LLM provider with a fixed
  prompt template; feature disabled if no provider is configured. Output stored with the provider and model name.

## 5. Training pipeline

| Stage | Trains | Frozen | Data | Key hyperparameters |
|---|---|---|---|---|
| S0 — Baseline | MFCC(40)+Δ+ΔΔ → MLP / 1D-CNN (thesis & repo replica) | — | RAVDESS, CREMA-D | Adam 1e-3, 50 epochs |
| S1 — Unimodal | Each branch head + LoRA | Base encoder weights | Per-branch datasets | AdamW 2e-4 (LoRA) / 2e-5 (full), batch 16, 15 epochs, fp16, early stop on val macro-F1 |
| S2 — Reliability | REM MLPs | All encoders & heads | Clean + corrupted multimodal data | AdamW 1e-3, 10 epochs |
| S3 — Fusion | Fusion + output heads (+ optional last-layer unfreeze) | Encoders, REM | IEMOCAP, MELD, CMU-MOSEI, RAVDESS-AV with modality dropout & corruption | AdamW 1e-4, cosine schedule, 30 epochs |
| S4 — Calibration | Temperature T, thresholds τ | Everything | Validation split | LBFGS on NLL |

All runs: fixed seeds, Hydra configs committed, MLflow logging of params, metrics, confusion
matrices, and artifacts. The registered model bundle contains ONNX graphs, label map,
calibration params, thresholds and a model card.

## 6. Evaluation protocol and acceptance criteria

### 6.1 Metrics

| Category | Metrics |
|---|---|
| Classification | Weighted accuracy (WA), unweighted accuracy (UA), macro-F1, confusion matrix |
| Dimensional | Concordance Correlation Coefficient (CCC) for valence and arousal |
| Calibration | ECE (15 bins), reliability diagram, NLL, Brier score |
| Selective prediction | Risk–coverage curve, AURC, accuracy @ 80% coverage |
| Reliability module | AUROC of r_m for predicting unimodal correctness |
| Robustness | Macro-F1 vs. corruption severity (0–4) per modality; missing-modality matrix |
| Fairness | Macro-F1 per gender and per dataset |
| Efficiency | Latency p50/p95 per stage, GPU memory, throughput |

### 6.2 Experiment matrix

| Exp | System |
|---|---|
| E0 | Thesis replica: MFCC + MLP (audio) |
| E1 | Repo replica: LibROSA + 1D-CNN (audio) |
| E2–E4 | Audio / Text / Visual branches alone |
| E5 | Static equal-weight late fusion |
| E6 | F1 reliability-weighted late fusion |
| E7 | F2 gated fusion |
| **E8** | **F3 RG-CMT (full RAMEI)** |
| E9–E11 | Ablations of E8: −REM, −modality dropout, −calibration |

### 6.3 Acceptance criteria

| ID | Criterion | Traces to |
|---|---|---|
| AC-01 | E8 macro-F1 ≥ E0 + 10 points on the same split | NFR-03 |
| AC-02 | E8 relative macro-F1 drop ≤ 10% with any single modality missing | NFR-04 |
| AC-03 | E8 ECE ≤ 0.05 | NFR-05 |
| AC-04 | REM AUROC ≥ 0.75 for every modality | FR-09 |
| AC-05 | E8 outperforms E5 under severity-3 corruption of any modality | FR-10 |
| AC-06 | Live p95 latency ≤ 1.5 s (GPU) / 3 s (CPU) per segment | NFR-01 |

## 7. Backend specification

### 7.1 Service responsibilities

| Module | Responsibility |
|---|---|
| `api` | HTTP/WS routing, request validation, auth dependencies |
| `core` | Settings, security (JWT, hashing), logging, exceptions |
| `services` | Business logic: sessions, uploads, alerts, summaries, exports |
| `pipeline` | Inference orchestration: preprocess → branches → REM → fusion → postprocess |
| `inference` | Model registry loading, ONNX Runtime sessions, batching |
| `workers` | Celery tasks for offline file analysis |
| `db` | SQLAlchemy models, repositories, unit of work |
| `schemas` | Pydantic request/response models |

### 7.2 REST API (v1)

Base path: `/api/v1`. JSON bodies. Auth: `Authorization: Bearer <JWT>` except health and login.

| Method | Path | Description | Roles |
|---|---|---|---|
| GET | `/health` | Liveness/readiness, model status | Public |
| POST | `/auth/login` | Email + password → access & refresh tokens | Public |
| POST | `/auth/refresh` | Refresh access token | Any |
| GET | `/users/me` | Current user profile | Any |
| POST | `/sessions` | Create session `{mode: live\|upload, title, consent: true}` | Agent+ |
| GET | `/sessions` | List sessions (pagination, filters) | Agent+ |
| GET | `/sessions/{id}` | Session metadata and status | Owner / Analyst+ |
| DELETE | `/sessions/{id}` | Delete session and all derived data | Owner / Admin |
| POST | `/sessions/{id}/media` | Multipart upload; enqueues analysis job | Owner |
| GET | `/sessions/{id}/segments` | Per-segment results (paginated) | Owner / Analyst+ |
| GET | `/sessions/{id}/timeline` | Downsampled timeline series | Owner / Analyst+ |
| GET | `/sessions/{id}/alerts` | Alerts for session | Owner / Analyst+ |
| GET | `/sessions/{id}/summary` | LLM summary (if generated) | Owner / Analyst+ |
| GET | `/sessions/{id}/export?format=json\|csv` | Export results | Owner / Analyst+ |
| POST | `/predict` | Stateless single-clip analysis (≤ 30 s) | Integrator |
| GET | `/models` | Active and available model versions | Admin |
| PUT | `/admin/settings` | Retention, raw-storage, alert rules | Admin |
| GET | `/admin/audit-logs` | Audit trail | Admin |

**Segment result schema (response)**

```json
{
  "segment_id": "uuid",
  "index": 12,
  "start_ms": 36400,
  "end_ms": 39150,
  "transcript": "I have called three times already",
  "emotion": "angry",
  "confidence": 0.87,
  "abstained": false,
  "probabilities": {"neutral": 0.03, "happy": 0.01, "sad": 0.05, "angry": 0.87, "fear": 0.02, "disgust": 0.01, "surprise": 0.01},
  "valence": -0.62,
  "arousal": 0.78,
  "conflict": false,
  "modalities": {
    "audio":  {"available": true, "reliability": 0.31, "weight": 0.12, "emotion": "neutral", "confidence": 0.41},
    "text":   {"available": true, "reliability": 0.85, "weight": 0.40, "emotion": "angry",   "confidence": 0.79},
    "visual": {"available": true, "reliability": 0.90, "weight": 0.48, "emotion": "angry",   "confidence": 0.88}
  },
  "explanation": ["Audio down-weighted: low SNR (4 dB)", "Face and words agree on anger"],
  "model_version": "ramei-rgcmt-1.0.0"
}
```

**Error format** (RFC 9457 Problem Details):
`{"type": "...", "title": "...", "status": 422, "detail": "...", "instance": "/api/v1/..."}`

### 7.3 WebSocket streaming API

Endpoint: `wss://<host>/api/v1/stream/{session_id}?token=<JWT>`

| Direction | Message | Payload |
|---|---|---|
| C → S | `start` | `{type, sample_rate: 16000, video: true\|false}` |
| C → S | `audio_chunk` | binary frame: header byte `0x01` + PCM16 mono, 200–500 ms |
| C → S | `video_frame` | binary frame: header byte `0x02` + JPEG (≤ 640×480, ≤ 8 fps) |
| C → S | `stop` | `{type}` |
| S → C | `segment_result` | Segment result schema (§7.2) |
| S → C | `partial_transcript` | `{type, text, start_ms}` |
| S → C | `alert` | `{type, rule, severity, start_ms, message}` |
| S → C | `status` | `{type, state: ready\|processing\|degraded\|closed, detail}` |
| S → C | `error` | Problem-details object |

Back-pressure: server drops video frames before audio when the queue exceeds 2 s of media and
emits `status: degraded`.

### 7.4 Offline job lifecycle

`pending → uploading → queued → processing → completed | failed` stored in `sessions.status`;
progress percentage published on Redis and exposed via `GET /sessions/{id}`.

## 8. Database specification

PostgreSQL 16; UUID primary keys (`gen_random_uuid()`); `timestamptz` everywhere; soft FK
cascades on session deletion. Full DDL: [`database/schema/schema.sql`](../database/schema/schema.sql).

| Table | Purpose | Key columns |
|---|---|---|
| `organizations` | Tenant | id, name, settings (JSONB) |
| `users` | Accounts | id, org_id, email (unique), password_hash, role, is_active |
| `consents` | Consent records | id, user_id, session_id, consent_text_version, granted_at, ip_hash |
| `model_versions` | Registered models | id, name, version, artifact_uri, metrics (JSONB), is_active |
| `sessions` | Analysis sessions | id, org_id, user_id, mode, title, status, progress, model_version_id, duration_ms, created_at |
| `media_assets` | Optional raw media | id, session_id, kind, storage_uri, size_bytes, sha256, expires_at |
| `segments` | Fused per-segment result | id, session_id, idx, start_ms, end_ms, transcript, emotion, confidence, abstained, valence, arousal, conflict, probabilities (JSONB), explanation (JSONB) |
| `modality_predictions` | Per-modality detail | id, segment_id, modality, available, reliability, weight, emotion, confidence, probabilities (JSONB), quality (JSONB) |
| `alerts` | Raised alerts | id, session_id, rule, severity, start_ms, end_ms, message, acknowledged |
| `session_summaries` | LLM summaries | id, session_id, provider, model, content, created_at |
| `audit_logs` | Security trail | id, org_id, user_id, action, resource, resource_id, metadata (JSONB), created_at |

Indexes: `segments(session_id, idx)`, `modality_predictions(segment_id)`, `sessions(org_id, created_at DESC)`,
`alerts(session_id, start_ms)`, `audit_logs(org_id, created_at DESC)`.

Retention: a scheduled job deletes `media_assets` past `expires_at` and sessions older than the
organization's retention period.

## 9. Frontend specification

| Area | Requirement |
|---|---|
| Framework | React 18 single-page app built with Vite 5, TypeScript strict mode |
| Routing | React Router 6; route-level code splitting (`React.lazy`); protected routes by role |
| Structure | App shell in `app/`, route screens in `pages/`, feature logic in `features/` (`live-session`, `session-report`, …) |
| Configuration | Build-time env vars prefixed `VITE_` (`VITE_API_URL`, `VITE_WS_URL`); no secrets in the bundle |
| Dev server | Vite dev server proxies `/api` (HTTP + WebSocket) to the backend to avoid CORS in development |
| Production | `vite build` → static assets served by the Nginx reverse proxy with SPA fallback to `index.html` |
| Data | TanStack Query for REST; a typed WebSocket client hook for streaming |
| Media capture | `getUserMedia` → AudioWorklet producing PCM16 @ 16 kHz; canvas JPEG frames @ 6 fps |
| Charts | Recharts: emotion timeline (stacked area), valence/arousal line, reliability bars |
| Accessibility | WCAG 2.1 AA; color-blind-safe emotion palette with text labels; keyboard navigation |
| Theming | Light/dark via CSS variables |
| API types | Generated from backend OpenAPI (`openapi-typescript`) — no hand-written duplicates |
| Auth | Access token in memory, refresh token in httpOnly secure cookie |

## 10. Performance budgets (live mode, per 3 s segment)

| Stage | GPU budget | CPU (ONNX) budget |
|---|---|---|
| VAD + preprocessing | 30 ms | 60 ms |
| Audio branch | 60 ms | 400 ms |
| ASR (whisper small.en) | 250 ms | 1200 ms |
| Text branch | 20 ms | 120 ms |
| Face detection + visual branch | 120 ms | 700 ms |
| REM + fusion + calibration | 5 ms | 20 ms |
| Network + serialization | 100 ms | 100 ms |
| **Total (branches in parallel)** | **≤ 1.5 s p95** | **≤ 3 s p95** |

## 11. Security and privacy requirements

| ID | Requirement |
|---|---|
| SEC-01 | TLS 1.2+ for all external traffic, including WebSockets |
| SEC-02 | Passwords hashed with Argon2id |
| SEC-03 | JWT access tokens (15 min) + rotating refresh tokens (7 days) |
| SEC-04 | RBAC enforced in API dependencies; row-level ownership checks per organization |
| SEC-05 | Upload validation: MIME sniffing, extension allow-list, size/duration limits, FFmpeg probe |
| SEC-06 | Rate limiting per user and IP (Redis token bucket) |
| SEC-07 | Secrets only via environment variables / secret store; never committed |
| SEC-08 | Raw media storage disabled by default; when enabled, encrypted at rest with expiry |
| SEC-09 | Audit log for login, export, delete, settings changes |
| SEC-10 | Dependency scanning (pip-audit, npm audit) and container scanning in CI |
| SEC-11 | No PII in logs; transcripts excluded from application logs |

## 12. Testing strategy

| Level | Scope | Tools | Gate |
|---|---|---|---|
| Unit | Services, REM, fusion math, label mapping, schema validation | pytest, Vitest | ≥ 80% backend coverage |
| Model tests | Shape/dtype contracts, missing-modality behavior, determinism | pytest | Required |
| Integration | API + DB + Redis via test containers | pytest + httpx | Required |
| Contract | OpenAPI schema diff; frontend types regenerate cleanly | openapi-typescript | Required |
| Frontend unit | Components, hooks, stores | Vitest + React Testing Library | Required |
| E2E | Login → upload → report; live session happy path | Playwright | Pre-release |
| Load | 20 concurrent live sessions | Locust | Pre-release |
| ML regression | Evaluation suite on frozen test set; fail if macro-F1 drops > 1 point | Custom + MLflow | On model change |

## 13. DevOps and environments

| Environment | Purpose | Deployment |
|---|---|---|
| Local | Development | `docker compose up` (Postgres, Redis, MinIO, backend, worker, frontend) |
| CI | Verification | GitHub Actions: lint (ruff, eslint), type-check (mypy, tsc), tests, build images |
| Staging | Demo / UAT | Single VM with GPU, Docker Compose, Caddy/Nginx TLS |
| Production (optional) | Pilot | Container orchestration with separate GPU inference workers |

Branching: trunk-based, short-lived feature branches, PR review required, Conventional Commits,
semantic versioning for both application and model bundles.

## 14. Observability

- **Logs:** JSON structured logs with `request_id`, `session_id`, `model_version`; no transcript text.
- **Metrics:** request latency, per-stage inference latency, queue depth, active streams,
  abstention rate, mean reliability per modality, error counts.
- **Drift signals:** rolling distribution of predicted emotions and reliability scores; alert on large shifts.
- **Dashboards:** Grafana board for API health and model behavior.

## 15. Coding standards

| Area | Standard |
|---|---|
| Python | ruff (lint + format), mypy strict on `backend/app` and `ml/src`, docstrings on public functions |
| TypeScript | ESLint + Prettier, `strict: true`, no `any` in feature code |
| SQL | Migrations only via Alembic; no manual schema edits |
| Commits | Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` …) |
| Configs | All tunables in Hydra (ML) or environment-based settings (backend) |

## 16. Repository layout

```
AD-INFOCOMM/
├── docs/        # PRD, TRD, architecture, ADRs, diagrams
├── frontend/    # React + Vite + TypeScript dashboard
├── backend/     # FastAPI service, inference pipeline, workers
├── database/    # Schema, migrations, seeds, ER docs
├── ml/          # Research & training: data, models, experiments, export
├── infra/       # Dockerfiles, reverse proxy, monitoring
└── scripts/     # Developer utilities
```

Detailed tree: [03-System-Architecture.md §11](03-System-Architecture.md#11-repository-mapping).

## 17. Traceability matrix

| PRD requirement | Technical component(s) |
|---|---|
| FR-01, FR-02 | Frontend media capture; WS API §7.3; upload endpoint §7.2; SEC-05 |
| FR-03, FR-04 | `sessions`, `consents` tables; session service |
| FR-05 – FR-07 | Preprocessing §3.4; pipeline module |
| FR-08 | Branches §4.1–4.3 |
| FR-09 | REM §4.4; AC-04 |
| FR-10, FR-11 | Fusion §4.5; modality mask; AC-02, AC-05 |
| FR-12, FR-13 | Output heads §4.5 |
| FR-14, FR-15 | Calibration & abstention §4.6; AC-03 |
| FR-16 | Temporal smoothing §4.7 |
| FR-17, FR-18 | Explainability §4.8 |
| FR-19 – FR-22 | Session intelligence §4.9; timeline/alerts/summary/export endpoints |
| FR-23 | REST §7.2, WS §7.3 |
| FR-24, FR-25 | SEC-02–04; admin endpoints |
| FR-26 | `model_versions` table; `model_version` in every result |
| FR-27 | Corruption module §3.5 exposed via sandbox |
| NFR-01, NFR-02 | Performance budgets §10; ONNX Runtime; load tests |
| NFR-07, NFR-08 | Security §11 |
| NFR-11, NFR-12 | Testing §12; MLflow + Hydra §5 |
