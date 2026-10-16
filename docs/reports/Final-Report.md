---
title: "RAMEI — Reliability-Aware Adaptive Multimodal Emotion Intelligence"
subtitle: "Internship Project Report — AD-INFOCOMM"
author: "[Student name]  ·  [Roll / ID number]"
guide: "[Guide name], [Designation]"
institution: "[Institution / Department]"
date: "[Submission date]"
---

# Abstract

Speech emotion recognition systems built on a single signal are brittle: they fail when the
voice is noisy or flat and they give no indication of when they are wrong. This project
started from an audio-only baseline (MFCC features with an MLP or 1D-CNN) and evolved it into
RAMEI, a multimodal system that recognizes emotion from voice, words and face together,
estimates how reliable each signal is for every segment, and shifts trust toward the reliable
ones. Every prediction carries a calibrated confidence, an explicit "uncertain" outcome instead
of a forced guess, and a plain-language explanation.

Re-evaluating the baseline with a speaker-independent protocol showed that its reported ~70%
accuracy depends on speaker leakage; on unseen speakers it reaches a macro-F1 of 0.51-0.55.
A WavLM-based audio branch raises this to 0.69-0.70 (+17 to +19 points), a DeBERTa text
branch reaches a weighted-F1 of 0.59 on MELD, and a ViT face branch completes the three
modalities. The reliability-gated fusion model, evaluated with 5-fold speaker-independent
cross-validation on CREMA-D (91 actors) under clean, noisy and missing-modality conditions,
[PENDING M2: headline fused result and robustness numbers].

The models are delivered as a working platform: a FastAPI service with authentication,
role-based access, file upload and live WebSocket streaming, and a React dashboard with
session timelines, per-segment explanations, alerts and summaries. The project is fully
reproducible from the public repository, with automated tests and continuous integration.

# 1. Introduction

## 1.1 Motivation

Organizations that talk to people over voice and video — support centres, e-learning,
telehealth — benefit from understanding the emotional state of a conversation as it happens.
Real conditions are hostile to any single signal: background noise masks the voice, scripted
or polite speech hides emotion in words, and cameras are often off, dark or pointed away.
A useful system must therefore combine signals and, just as importantly, know when not to
trust one of them.

## 1.2 The inherited baseline

The internship started from two audio-only artefacts:

- A 2021 thesis, *Speech Emotion Recognition using Machine Learning*, proposing MFCC, pitch
  and energy features with an MLP classifier.
- The open-source *Speech-Emotion-Analyzer* repository: a Keras 1D-CNN on LibROSA features
  over RAVDESS and SAVEE, reporting about 70% accuracy.

| # | Gap in the baseline | Consequence |
|---|---|---|
| G1 | Audio is the only modality | Fails when the voice is unclear, flat or missing |
| G2 | No reliability or quality awareness | Confidently wrong on noisy or silent input |
| G3 | Uncalibrated confidence | Scores cannot be used for decisions |
| G4 | No handling of missing inputs | Breaks when the camera or microphone is off |
| G5 | One label per file | No timeline, valence or arousal, or insight |
| G6 | Small acted datasets, unclear splits | Results do not transfer to new speakers |
| G7 | Notebook only | Not usable by an end user or another system |

## 1.3 Objectives

1. Establish an honest, reproducible baseline with speaker-independent evaluation.
2. Build strong unimodal branches for audio, text and face.
3. Estimate per-modality reliability and fuse adaptively, with calibrated confidence and
   abstention (target: ECE ≤ 0.05; ≤ 10% relative drop with one modality degraded or missing).
4. Deliver the models as a secure, usable platform: API, live streaming and dashboard.
5. Document every decision and make every result reproducible.

## 1.4 Contributions

- A leakage analysis showing that the baseline's headline accuracy is inflated by 15-21 points.
- Audio, text and visual branches built on pretrained encoders with parameter-efficient tuning.
- A reliability-gated fusion model trained with corruption and modality-dropout augmentation,
  plus ablations isolating the effect of reliability.
- An end-to-end platform (FastAPI + React) with explanations, alerts and summaries.
- Engineering practice: architecture decision records, CI, 60 automated tests, container images.

## 1.5 Report structure

Section 2 describes the data and evaluation protocol, Section 3 the models, Section 4 the
platform, and Section 5 the results. Section 6 maps requirements to what was delivered,
Section 7 discusses limitations and ethics, and Section 8 concludes. The appendices cover
reproduction steps and the repository layout.

# 2. Data and evaluation protocol

## 2.1 Datasets

| Dataset | Content | Used for |
|---|---|---|
| RAVDESS | 24 actors (12 F / 12 M), 1,440 acted speech clips | Baselines and audio branch (M0, M1) |
| CREMA-D | 91 actors, 7,442 clips with audio and video | Visual branch and fusion (M2) |
| MELD | Multi-party TV dialogue, ~13k utterances, official splits | Text branch (M1) |

All datasets are mapped to one 7-class label space: neutral, happy, sad, angry, fear, disgust
and surprise (RAVDESS "calm" is merged into neutral).

## 2.2 Speaker-independent evaluation

Emotion corpora contain many clips per speaker. If the same speaker appears in training and
test data, a model can recognize the voice rather than the emotion. All reported results
therefore use **speaker-independent splits**: RAVDESS uses 6 gender-balanced folds of 4 actors
each, and CREMA-D uses 5 folds over its 91 actors. Every speaker is tested exactly once, and
results are reported as the mean ± standard deviation across folds.

## 2.3 Metrics

- **Macro-F1** (primary): the unweighted mean of per-class F1, robust to class imbalance.
- Weighted accuracy (WA) and unweighted accuracy (UA).
- **Expected Calibration Error (ECE)**: the gap between stated confidence and observed accuracy.
- Abstention rate and accuracy on the predictions the model does commit to.
- Per-gender macro-F1 as a fairness check.

## 2.4 Robustness scenarios

The fusion model is tested under twelve conditions built from frozen-encoder features:
clean input; audio noise at 10 dB and 0 dB SNR (with speech recognition re-run on the noisy
audio); blurred, darkened and occluded faces; 50% of transcript words deleted; camera off;
audio missing; noisy room with camera off; and audio-only and visual-only.

# 3. Methods

## 3.1 Baselines (M0)

- **E0** reproduces the thesis: MFCC with deltas, pitch, energy, zero-crossing rate and
  spectral centroid statistics, followed by an MLP.
- **E1** reproduces the repository's 1D-CNN on MFCC sequences.

Both use 16 kHz mono audio with silence trimming and loudness normalization.

## 3.2 Audio branch

The audio branch uses WavLM-Base+, a self-supervised speech encoder. A learned softmax
weighting over its 13 hidden layers is followed by attentive pooling and a classification
head. Two variants were compared: LoRA fine-tuning (rank 8) of the attention projections
(E2), and a fully frozen encoder (E2a).

## 3.3 Text branch

Speech is transcribed with faster-whisper (small.en). DeBERTa-v3-base with LoRA is fine-tuned
on MELD, which consists of natural conversational text rather than acted sentences.
Temperature scaling is fitted on the validation split.

## 3.4 Visual branch

MediaPipe face detection produces square crops with a 25% margin at 224×224 pixels, for 16
frames per clip. Each frame also yields quality features: detection confidence, face area,
blur (Laplacian variance) and brightness. A frozen ViT-B/16 facial-expression model encodes
each frame, and a small temporal transformer with a [CLS] token aggregates the frames.

## 3.5 Reliability estimation

For each modality *m*, a small network predicts a reliability score *r_m* in [0, 1]. Its
inputs are the normalized quality features (SNR and voice activity for audio; speech
recognition log-probability, no-speech probability, compression ratio and word count for
text; face presence, confidence, size, blur and brightness for visual) and the branch's own
uncertainty (entropy, top probability and top-2 margin).

The score is trained with a binary target, "was this branch's prediction correct?", so *r_m*
remains an interpretable probability rather than an unconstrained gate.

## 3.6 Reliability-gated fusion

The three branch embeddings become tokens for a 2-layer cross-modal transformer with a [CLS]
token (ADR-0001). Attention to modality *k* receives an additive bias β·log(*r_k* + ε), where
β is learned. A missing modality is masked out entirely, so the model works with any
non-empty subset of inputs.

The training loss is:

> L = CE(fused) + λ_aux · Σ CE(branch) + λ_rel · BCE(r_m, branch correct)

with label smoothing of 0.1 and class weighting.

Each training example draws, per modality, either the clean version (probability 0.4) or a
random corruption, and drops a modality with probability 0.2, always keeping at least one.

## 3.7 Calibration, abstention and explanation

A temperature is fitted per fold on validation speakers. At inference, the system abstains
and labels the segment "uncertain" when the calibrated top probability falls below a
threshold. It flags a **conflict** when two reliable modalities disagree. Explanations are
generated from the reliability inputs, for example "Voice down-weighted: background noise
(SNR ≈ 4 dB)", "Face is poorly lit" or "No words recognized", together with the modality
that carried the most weight.

Valence and arousal are estimated as the probability-weighted position of each emotion on
the circumplex, because CREMA-D and MELD carry no valence or arousal labels.

## 3.8 Ablations

| ID | Variant | Question answered |
|---|---|---|
| E5 | Equal-weight average of branch probabilities | Does learned fusion beat simple averaging? |
| E6 | Reliability-weighted late fusion | Is reliability alone enough? |
| E8 | Full RAMEI (reliability-gated transformer, robust training) | Proposed model |
| E9 | E8 without the reliability bias | What does reliability add? |
| E10 | E8 trained on clean, complete inputs only | What does robust training add? |

# 4. System

## 4.1 Architecture

```
Browser (React + Vite) ──HTTPS / WebSocket──► FastAPI
                                                ├─ auth, sessions, uploads, live stream
                                                ├─ Inference engine (versioned model bundle)
                                                │    speech segments → [WavLM | Whisper+DeBERTa | Face+ViT]
                                                │    → reliability per modality → reliability-gated fusion
                                                │    → calibration / abstention → explanation
                                                └─ SQLite (dev) / PostgreSQL (prod)
```

The ML package and the backend meet at one contract: a **versioned model bundle** containing
a manifest, weights, the calibration temperature, abstention threshold and cross-validation
metrics. The backend never imports training code or loose checkpoints.

## 4.2 Backend

- **API:** REST under `/api/v1` with OpenAPI documentation and RFC 9457 problem-details errors.
- **Endpoints:** authentication; sessions (create, list, detail, delete, upload, segments,
  alerts, export to JSON or CSV); dashboard statistics; health and model status.
- **Live mode:** a WebSocket receives audio chunks and JPEG frames, analyses 3-second windows
  and streams results back. Video frames are dropped first if analysis falls behind.
- **Upload mode:** media is validated (format, size of at most 200 MB, duration of at most
  60 minutes), split into speech segments, analysed by a worker, then deleted. Raw media is not
  stored by default.
- **Insights:** rule-based alerts (sustained anger of 20 s or more, negative valence for 30 s
  or more, three or more consecutive uncertain segments) and a data-driven session summary.
  An optional LLM narrative is available (Section 4.4).

## 4.3 Frontend

The React and TypeScript dashboard has six screens: overview, sessions, upload, live, session
report and system. The session report shows an emotion and valence timeline, a transcript, and
a segment inspector that displays per-modality predictions, reliability, fusion weights and
explanations.

The interface supports light and dark themes, splits code by route, and every analysis
requires explicit consent.

## 4.4 Security and privacy

| Control | Implementation |
|---|---|
| Authentication | JWT access tokens; Argon2 password hashing |
| Authorization | Roles (agent, analyst, admin) with organization isolation, covered by tests |
| Abuse protection | Token-bucket rate limit on login |
| Input handling | Server-generated upload names, streamed size cap, media validation |
| Transport and headers | nosniff, frame denial, no-referrer, no-store, HSTS outside local; CSP at the proxy |
| Secrets | Production refuses to start without a JWT secret of 32 or more characters |
| Privacy | Uploads deleted after analysis; consent recorded per session; the optional LLM narrative receives only aggregate statistics, never transcripts or media |
| Logging | Tokens are kept out of access logs (the proxy logs paths only) |

## 4.5 Deployment

The project provides Docker images for the API (CPU PyTorch, non-root user, health check) and
the dashboard (nginx). nginx serves the single-page app and proxies `/api`, including
WebSocket upgrades. A Compose file wires the stack together, with PostgreSQL as an optional
profile.

Development used a local-first setup recorded in ADR-0008: SQLite in place of PostgreSQL,
in-process workers in place of Celery, and PyTorch in place of ONNX serving. Each of these
keeps the same interfaces, so the production path stays open.

## 4.6 Engineering practice

- **Documentation:** PRD, TRD, system architecture document, architecture decision records,
  and a report per milestone.
- **Tests:** 60 automated tests (37 ML, 18 backend, 5 frontend) that run without datasets.
- **CI:** GitHub Actions runs linting, type-checking, tests and the frontend build on every push.

# 5. Results

## 5.1 The honest baseline (M0)

| Experiment | WA | UA | Macro-F1 | ECE |
|---|---|---|---|---|
| E0: thesis MLP | 0.526 ± 0.070 | 0.516 ± 0.072 | **0.512 ± 0.076** | 0.279 |
| E1: repo 1D-CNN | 0.557 ± 0.046 | 0.550 ± 0.053 | **0.549 ± 0.050** | 0.200 |

*RAVDESS, 6-fold speaker-independent cross-validation.*

| Macro-F1 | Speaker-independent | Random split (leaky) | Inflation |
|---|---|---|---|
| E0 | 0.550 | 0.696 | +14.6 points |
| E1 | 0.538 | 0.750 | +21.2 points |

*Same models and data; only the split differs.*

The baseline's reported ~70% is reproduced only when test speakers also appear in training.
On unseen speakers the honest figure is 0.51-0.55, and confidence is badly miscalibrated
(ECE 0.20-0.28).

## 5.2 Unimodal branches (M1)

| Branch | Data / protocol | Macro-F1 | Other |
|---|---|---|---|
| Audio, E2: WavLM + LoRA | RAVDESS, 6-fold CV | **0.686 ± 0.028** | ECE 0.083 |
| Audio, E2a: WavLM frozen | RAVDESS, 6-fold CV | **0.701 ± 0.062** | ECE 0.079 calibrated |
| Text: DeBERTa-v3 + LoRA | MELD test (2,610) | 0.448 | Weighted-F1 0.591; ECE 0.122 → 0.082 |

- The audio branch gains +17 to +19 points over the thesis baseline on unseen speakers. This
  meets the "baseline + 10 points" acceptance criterion with audio alone.
- LoRA and the frozen encoder are statistically tied on 24 actors, so fusion uses frozen
  features, which is cheaper and more stable.
- Text-only performance on MELD is in line with published text-only results. Fear and disgust
  (about 2% of the data each) remain hard.
- The face pipeline processed all 7,442 CREMA-D clips. Its visual-only results come from the
  fusion evaluation (Section 5.3).

## 5.3 Reliability-gated fusion (M2)

[PENDING M2: table of macro-F1 per experiment (E5, E6, E8, E9, E10, branch-only) per
scenario, 5-fold CV on CREMA-D]

[PENDING M2: robustness: relative drop from clean per scenario versus the NFR-04 target of
≤ 10%]

[PENDING M2: calibration: ECE raw and calibrated versus the NFR-05 target of ≤ 0.05;
abstention rate and selective accuracy]

[PENDING M2: fairness: female versus male macro-F1 gap versus the NFR-09 target of
≤ 5 points]

[PENDING M2: reliability analysis: mean r_m per scenario showing that trust moves away from
the corrupted modality]

## 5.4 Platform performance

[PENDING: API load test on an idle machine: throughput, p50/p95/p99 latency, error rate]

[PENDING M2: per-segment inference latency on the RTX 3050 (GPU) and on CPU versus NFR-01]

# 6. Requirements traceability

| Requirement | Status | Where |
|---|---|---|
| FR-01/02: live capture and file upload | Done | Live and Upload pages; `/stream`, `/sessions/{id}/media` |
| FR-03/04: sessions, consent | Done | Sessions API; consent required on create |
| FR-05/06: segmentation, transcription | Done | Energy-based segmentation (ADR-0008); faster-whisper |
| FR-07/08: face tracking, per-modality prediction | Done | MediaPipe + ViT; branch outputs per segment |
| FR-09/10/11: reliability, adaptive fusion, any subset | Done | Reliability estimator; reliability-gated transformer with masking |
| FR-12/14: 7 emotions, calibrated confidence, abstention | Done | Temperature scaling and threshold |
| FR-13: valence / arousal | Partial | Circumplex estimate; no labelled V/A training data |
| FR-15: cross-modal conflict | Done | Engine conflict flag, shown in the inspector |
| FR-16: temporal smoothing | Not done | Planned |
| FR-17: weights and reasons | Done | Segment inspector |
| FR-18: saliency cues (Could) | Not done | Future work |
| FR-19/20: timeline, alerts | Done | Report timeline; alert rules with acknowledgement |
| FR-21: session summary | Done | Rule-based summary; optional LLM narrative |
| FR-22: JSON/CSV export | Done | `/sessions/{id}/export` |
| FR-23/24: REST + WebSocket, RBAC | Done | Versioned API; three roles with organization isolation |
| FR-25: retention settings | Partial | Stored per organization; no admin screen yet |
| FR-26: model version per result | Done | Stored on each session and segment |
| FR-27: research sandbox (Could) | Offline only | Corruption scenarios in the M2 evaluation |

# 7. Limitations and ethics

- **Acted data.** RAVDESS and CREMA-D are acted, and their actors read a few fixed sentences,
  so the text modality carries almost no emotion signal there. Real conversations will differ.
  IEMOCAP or MSP-Podcast (license required) would provide a fairer test.
- **Small speaker pools.** Fold-to-fold variance is 3-8 F1 points. Single-split numbers are
  unreliable, which is why everything is cross-validated.
- **Valence and arousal** are derived from class probabilities, not regressed against labels.
- **Local-first implementation choices** (ADR-0008) are adequate for a demo, not for
  multi-tenant production. The Docker images have not been built on the development machine.
- **Ethics.** Emotion recognition is probabilistic and culturally variable. RAMEI must not be
  used to make decisions about individuals (hiring, discipline, diagnosis) or to monitor people
  without their consent. The system defaults to recording consent, deleting raw media after
  analysis, abstaining when uncertain, and explaining every prediction.

# 8. Conclusion and future work

The project replaced an audio-only, leakage-inflated baseline with an honestly evaluated,
reliability-aware multimodal system and a working platform around it.
[PENDING M2: one-sentence fusion outcome.]

Future work:

1. Fine-tune encoders end-to-end, with more GPU memory.
2. Train on natural conversational data (IEMOCAP or MSP-Podcast), including a true
   valence/arousal head.
3. Add temporal smoothing and saliency cues.
4. Move to ONNX serving and Celery workers on PostgreSQL for production.
5. Build an admin screen for retention policy.

# Appendix A. Reproducing the results

```bash
# Environment (one virtualenv for ML and backend)
cd ml && python -m venv .venv && .venv/Scripts/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,audio,multimodal]" && pip install -e "../backend[dev,inference]"

# M0 baselines
python scripts/download_ravdess.py
python -m ramei_ml.training.cv                     # E0, 6-fold
python -m ramei_ml.training.cv experiment=e1_cnn   # E1, 6-fold

# M1 branches
python -m ramei_ml.training.audio_branch
python -m ramei_ml.training.text_branch

# M2 fusion and model bundle
python scripts/download_crema_d.py && python scripts/download_crema_d_video.py
python scripts/run_m2.py

# Platform
cd ../backend && python -m app.cli create-user --email admin@ramei.local --role admin
uvicorn app.main:app --port 8000
cd ../frontend && npm install && npm run dev
```

# Appendix B. Repository layout

| Folder | Contents |
|---|---|
| `docs/` | PRD, TRD, architecture, ADRs, milestone reports, this report |
| `ml/` | Data pipelines, branches, fusion, evaluation, bundle export, tests |
| `backend/` | FastAPI service, inference engine, analysis pipeline, tests |
| `frontend/` | React + Vite + TypeScript dashboard, tests |
| `database/` | Reference schema |
| `infra/` | Dockerfiles and nginx configuration |
| `scripts/` | API load test, report builder |
