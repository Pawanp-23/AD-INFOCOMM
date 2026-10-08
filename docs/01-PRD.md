# Product Requirements Document (PRD)

## RAMEI — Reliability-Aware Adaptive Multimodal Emotion Intelligence

| Field | Value |
|---|---|
| Document ID | RAMEI-PRD-001 |
| Version | 1.0 |
| Status | Draft for review |
| Date | 2026-10-09 |
| Owner | Product / Project Lead — AD-INFOCOMM internship team |
| Reviewers | Internship guide, ML lead, Backend lead, Frontend lead |
| Related documents | [02-TRD.md](02-TRD.md), [03-System-Architecture.md](03-System-Architecture.md) |

### Revision history

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-10-09 | Project team | Initial PRD |

---

## 1. Executive summary

RAMEI is a real-time emotion intelligence platform that understands how a person feels by
combining three signals — **voice (audio), words (text), and face (video)** — instead of
relying on a single one.

Its defining feature is that it **knows when to trust each signal**. For every prediction,
RAMEI estimates the reliability of each modality (e.g. noisy audio, a dark or occluded face,
a low-confidence transcript) and **adaptively re-weights** its fusion toward the trustworthy
sources. When nothing is trustworthy enough, it **abstains** instead of guessing. Every output
carries a calibrated confidence score and a human-readable explanation.

The product ships as a streaming API and a web dashboard, with call-center / video-call
emotion analytics as the primary use case.

## 2. Background and problem statement

### 2.1 Baseline we inherit

The internship baseline consists of:

1. **Thesis — "Speech Emotion Recognition using Machine Learning" (2021).** Proposes MFCC +
   pitch + energy features with an MLP classifier. The appendix code only visualizes
   waveforms/spectrograms; it never trains a classifier and reports no accuracy.
2. **GitHub — MiteshPuthran/Speech-Emotion-Analyzer.** A Keras 1D-CNN on LibROSA features over
   RAVDESS + SAVEE, reporting ~70% emotion accuracy in notebooks.

Both are **single-modal (audio only)**.

### 2.2 Gaps in the baseline

| # | Gap | Consequence |
|---|---|---|
| G1 | Audio is the only modality | Fails whenever the voice is unclear, flat, or missing |
| G2 | No reliability / quality awareness | Confidently wrong on noisy or silent input |
| G3 | Uncalibrated softmax confidence | Scores cannot be trusted for decisions |
| G4 | No handling of missing inputs | Breaks when the camera or mic is off |
| G5 | One label per file | No temporal view, no valence/arousal, no insight |
| G6 | Acted, small datasets; unclear splits | Results do not generalize to real users |
| G7 | Notebook-only | Not usable by any end user or system |

### 2.3 Problem statement

> Organizations that interact with people over voice and video (support centers, e-learning,
> telehealth) need to understand emotional state in real time. Existing single-modal emotion
> recognizers are brittle in real conditions and give no signal about when they are wrong.
> We need an emotion intelligence system that fuses multiple modalities, measures how
> reliable each one is at every moment, adapts its decision accordingly, and communicates
> calibrated, explainable results.

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal |
|---|---|
| G-1 | Recognize emotion from audio, text and video, individually and fused |
| G-2 | Estimate per-modality reliability for every prediction |
| G-3 | Adaptively weight modalities by reliability; work with any subset of modalities |
| G-4 | Provide calibrated confidence and an abstain ("uncertain") outcome |
| G-5 | Explain every prediction (weights, quality reasons, salient cues) |
| G-6 | Deliver real-time streaming analysis and offline file analysis |
| G-7 | Produce session-level intelligence: timeline, escalation alerts, summary |
| G-8 | Demonstrate measurable improvement over the thesis/repo baseline |

### 3.2 Non-goals (this release)

- Clinical or medical diagnosis of mental-health conditions.
- Physiological sensors (EEG, heart rate, GSR).
- Identity recognition / face recognition of individuals.
- Languages other than English (multilingual is a roadmap item).
- Deception or lie detection.

## 4. Target users and personas

| Persona | Description | Primary need |
|---|---|---|
| **Support Supervisor** (primary) | Oversees customer-support agents and call quality | Spot escalating, frustrated callers and review calls quickly |
| **Support Agent** | Handles live calls | Live cue about customer emotion to adapt tone |
| **Analyst / Researcher** | Studies interaction quality, runs experiments | Exportable, trustworthy per-segment data |
| **Developer / Integrator** | Embeds emotion analytics into another product | Clean, documented API with stable contracts |
| **Administrator** | Runs the deployment | Manage users, retention, consent, model versions |

## 5. Use cases and user stories

### 5.1 Primary use cases

| ID | Use case |
|---|---|
| UC-1 | Live analysis of a voice or video call with a real-time emotion timeline |
| UC-2 | Upload a recorded call/video and receive a full analysis report |
| UC-3 | Review a past session: timeline, transcript with emotions, alerts, summary |
| UC-4 | Integrate RAMEI into a third-party app through the REST/WebSocket API |
| UC-5 | Compare model behavior under noise / missing modalities (research mode) |

### 5.2 User stories

| ID | As a… | I want to… | So that… | Priority |
|---|---|---|---|---|
| US-01 | Supervisor | see live emotion and confidence during a call | I can intervene before escalation | Must |
| US-02 | Supervisor | get an alert when anger/frustration persists | I don't miss critical moments | Must |
| US-03 | Supervisor | read an auto-generated session summary | I review calls in seconds, not minutes | Should |
| US-04 | Agent | see a simple live mood indicator | I can adapt my tone | Should |
| US-05 | Analyst | see which modality drove each prediction and why | I can trust and audit results | Must |
| US-06 | Analyst | export per-segment results as CSV/JSON | I can do my own analysis | Should |
| US-07 | Developer | send audio-only, video-only or both | it works with what my app has | Must |
| US-08 | Developer | receive "uncertain" instead of a guess | my app doesn't act on bad predictions | Must |
| US-09 | Admin | configure data retention and consent | we comply with privacy policies | Must |
| US-10 | Researcher | inject noise / drop modalities in a sandbox | I can demonstrate robustness | Could |

## 6. Functional requirements

Priority uses MoSCoW (Must / Should / Could / Won't-this-release).

### 6.1 Input and session management

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system shall accept live audio and/or video from the browser (mic/webcam) | Must |
| FR-02 | The system shall accept uploaded files (WAV, MP3, MP4, WEBM; ≤ 200 MB, ≤ 60 min) | Must |
| FR-03 | The system shall create, list, view and delete analysis sessions | Must |
| FR-04 | The system shall record user consent before any capture or upload | Must |

### 6.2 Processing and recognition

| ID | Requirement | Priority |
|---|---|---|
| FR-05 | The system shall segment speech into utterances using voice activity detection | Must |
| FR-06 | The system shall transcribe speech to text with word-level timestamps | Must |
| FR-07 | The system shall detect and track the primary face in video | Must |
| FR-08 | The system shall produce an emotion prediction per available modality | Must |
| FR-09 | The system shall compute a reliability score in [0, 1] per modality per segment | Must |
| FR-10 | The system shall fuse modalities with reliability-adaptive weights | Must |
| FR-11 | The system shall operate with any non-empty subset of modalities | Must |
| FR-12 | The system shall output one of 7 emotions: neutral, happy, sad, angry, fear, disgust, surprise | Must |
| FR-13 | The system shall output continuous valence and arousal in [-1, 1] | Should |
| FR-14 | The system shall output calibrated confidence and abstain ("uncertain") below threshold | Must |
| FR-15 | The system shall flag cross-modal conflict (e.g. positive words, angry tone) | Should |
| FR-16 | The system shall smooth predictions over time to avoid flicker | Should |

### 6.3 Explainability and intelligence

| ID | Requirement | Priority |
|---|---|---|
| FR-17 | Each prediction shall include fusion weights and reliability reasons | Must |
| FR-18 | The system shall provide saliency cues (audio time regions, key words, face region) | Could |
| FR-19 | The system shall generate a session timeline of emotion, valence and arousal | Must |
| FR-20 | The system shall raise configurable alerts (e.g. anger > 20 s) | Should |
| FR-21 | The system shall generate a natural-language session summary | Should |
| FR-22 | The system shall export session results as JSON and CSV | Should |

### 6.4 Platform

| ID | Requirement | Priority |
|---|---|---|
| FR-23 | The system shall expose a versioned REST API and a WebSocket streaming API | Must |
| FR-24 | The system shall authenticate users and enforce role-based access (Admin, Analyst, Agent) | Must |
| FR-25 | Admins shall configure retention period and raw-media storage on/off | Must |
| FR-26 | The system shall record which model version produced each result | Must |
| FR-27 | A research sandbox shall let users inject noise/blur or drop modalities | Could |

## 7. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Latency | Live mode end-to-end ≤ 1.5 s per segment on GPU; ≤ 3 s on CPU (ONNX) |
| NFR-02 | Throughput | ≥ 20 concurrent live sessions per GPU worker |
| NFR-03 | Accuracy | Fused macro-F1 ≥ baseline audio-only + 10 points on the multimodal test set |
| NFR-04 | Robustness | ≤ 10% relative macro-F1 drop with one modality missing or heavily corrupted |
| NFR-05 | Calibration | Expected Calibration Error (ECE) ≤ 0.05 after calibration |
| NFR-06 | Availability | 99.5% monthly for the API (production deployment) |
| NFR-07 | Privacy | No raw media persisted by default; encryption in transit (TLS) and at rest |
| NFR-08 | Security | OWASP Top-10 mitigations; JWT auth; rate limiting |
| NFR-09 | Fairness | Report per-gender and per-dataset metrics; gap ≤ 5 F1 points targeted |
| NFR-10 | Usability | First analysis achievable in ≤ 3 clicks; WCAG 2.1 AA dashboard |
| NFR-11 | Maintainability | ≥ 80% unit-test coverage on backend services; typed code; CI on every PR |
| NFR-12 | Reproducibility | Every model traceable to config, data version and commit |
| NFR-13 | Portability | Entire stack runs with `docker compose up` on a single machine |

## 8. Success metrics (KPIs)

| Metric | Target | How measured |
|---|---|---|
| Macro-F1 (fused, 7-class) | ≥ 0.65 on IEMOCAP/MELD speaker-independent test | Evaluation suite |
| Improvement over thesis baseline | ≥ +10 F1 points | Same test split |
| Robustness drop (missing modality) | ≤ 10% relative | Stress-test suite |
| ECE | ≤ 0.05 | Calibration report |
| Selective accuracy at 80% coverage | ≥ 0.75 | Risk–coverage curve |
| Live latency (p95) | ≤ 1.5 s GPU | Load test |
| Supervisor review time per call | −50% vs. manual | User study |
| Alert precision (escalation) | ≥ 0.70 | Labeled pilot sessions |

## 9. User experience overview

| Screen | Purpose | Key elements |
|---|---|---|
| Login | Authentication | Email/password, role-based landing |
| Dashboard (Home) | Overview | Recent sessions, alert feed, KPI tiles |
| Live Session | Real-time analysis | Video/mic preview, current emotion + confidence, modality reliability bars, fusion weights, rolling timeline, live transcript |
| Upload | Offline analysis | Drag-and-drop, consent checkbox, progress status |
| Session Report | Review | Emotion & valence/arousal timeline, transcript colored by emotion, alerts, LLM summary, export |
| Explain Panel | Trust | Per-segment weights, quality reasons, saliency views |
| Research Sandbox | Demonstration | Noise/blur sliders, modality toggles, side-by-side baseline vs. RAMEI |
| Admin | Governance | Users & roles, retention, model versions, audit log |

## 10. Release plan

| Phase | Scope | Exit criteria |
|---|---|---|
| **M0 — Foundations** (Week 1) | Repo, docs, CI, data pipelines, thesis baseline replicated with honest split | Baseline metrics recorded |
| **M1 — Unimodal models** (Weeks 2–3) | Audio (WavLM), Text (Whisper + DeBERTa), Visual (ViT FER) branches | Each beats its own baseline |
| **M2 — Reliability + Fusion** (Weeks 4–5) | Reliability module, gated fusion, calibration, abstention | NFR-03/04/05 met offline |
| **M3 — Platform MVP** (Weeks 6–7) | FastAPI, WebSocket streaming, PostgreSQL, dashboard (Live, Upload, Report) | End-to-end demo works |
| **M4 — Intelligence + Hardening** (Week 8) | Alerts, LLM summary, explainability UI, sandbox, load tests, security review | All Must FRs pass acceptance |
| **M5 — Report & Demo** (Week 9) | Final evaluation, thesis/report, demo video | Guide sign-off |

## 11. Assumptions, constraints, dependencies

**Assumptions**
- English speech; one primary speaker in view per session.
- Access to at least one CUDA GPU for training (local, Colab, or Kaggle).
- Research datasets are used under their academic licenses.

**Constraints**
- Internship timeline of ~9 weeks.
- Some datasets (IEMOCAP, MSP-Podcast) require license requests; fall back to RAVDESS, CREMA-D,
  MELD, CMU-MOSEI if delayed.

**Dependencies**
- Pretrained models: WavLM / emotion2vec, Whisper, DeBERTa/RoBERTa, ViT/AffectNet-pretrained FER.
- Optional external LLM API for summaries (feature is disabled when no key is configured).

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset license delays | Medium | High | Start with open datasets; request licenses in Week 1 |
| Acted data does not generalize | High | Medium | Include natural data (MELD, CMU-MOSEI); cross-corpus evaluation |
| Insufficient GPU | Medium | High | LoRA fine-tuning, frozen encoders, mixed precision, Kaggle/Colab |
| Latency too high for live mode | Medium | Medium | ONNX export, quantization, smaller Whisper, frame sub-sampling |
| Fusion ignores a modality (collapse) | Medium | Medium | Modality dropout, reliability supervision, ablation monitoring |
| Bias across gender/accent | Medium | High | Per-group metrics, balanced sampling, documented limitations |
| Privacy concerns from users | Medium | High | Consent, no raw storage by default, retention controls |

## 13. Responsible AI, privacy and ethics

- **Consent-first:** no capture or upload without explicit, recorded consent.
- **Data minimization:** store derived predictions; raw media off by default.
- **No identity:** face data is used only for expression, never identity.
- **Honest uncertainty:** abstain rather than over-claim; display confidence everywhere.
- **Human in the loop:** RAMEI assists people; it does not make automated decisions about them.
- **Transparency:** publish a model card with datasets, metrics, per-group performance and limits.

## 14. Open questions

| # | Question | Owner |
|---|---|---|
| Q1 | Is the final deliverable a research report, a deployable product demo, or both? | Guide |
| Q2 | Which LLM provider (if any) is approved for session summaries? | Guide / Company |
| Q3 | Is on-premise deployment required, or is cloud acceptable? | Company |
| Q4 | Will labeled company call data be available for a pilot? | Company |

## 15. Glossary

| Term | Meaning |
|---|---|
| Modality | A type of input signal: audio, text, or video |
| Reliability score | Model-estimated probability that a modality's prediction is correct for this input |
| Adaptive fusion | Combining modalities with weights that change per input |
| Calibration | Making confidence scores match true accuracy |
| Abstain | Returning "uncertain" instead of a label |
| Valence / Arousal | Continuous emotion dimensions: negative↔positive, calm↔excited |
| ECE | Expected Calibration Error |
| VAD | Voice Activity Detection |
| Macro-F1 | F1 averaged equally across classes |
