# M0 Report — Foundations & Honest Baseline

| Field | Value |
|---|---|
| Milestone | M0 — Foundations (PRD §10) |
| Date | 2026-10-09 |
| Status | Complete |
| Experiments | E0 (thesis MLP), E1 (repo 1D-CNN) — TRD §6.2 |

## 1. What was delivered

| Item | Location |
|---|---|
| Unified 7-class label space + mappings for RAVDESS, CREMA-D, TESS, SAVEE | `ml/src/ramei_ml/data/labels.py` |
| Filename parsers (speaker id, gender, emotion, media type) | `ml/src/ramei_ml/data/parsers.py` |
| Speaker-independent, gender-stratified split manifests (+ leakage check) | `ml/src/ramei_ml/data/manifest.py`, `ml/data/manifests/` |
| Audio preprocessing (16 kHz mono, silence trim, loudness norm) | `ml/src/ramei_ml/features/audio.py` |
| Baseline features: MFCC+Δ+ΔΔ stats, pitch, energy, ZCR, centroid; MFCC sequences | `ml/src/ramei_ml/features/mfcc.py` |
| Metrics: WA, UA, macro-F1, ECE, NLL, Brier, confusion, per-group | `ml/src/ramei_ml/evaluation/metrics.py` |
| E0 thesis MLP, E1 repo-style 1D-CNN, trainer with early stopping | `ml/src/ramei_ml/models/baseline.py`, `training/` |
| Speaker-independent k-fold cross-validation | `ml/src/ramei_ml/training/cv.py` |
| Hydra experiment configs | `ml/configs/` |
| Reproducible RAVDESS download | `ml/scripts/download_ravdess.py` |
| 22 automated tests on synthetic audio (no dataset needed) | `ml/tests/` |
| CI: lint + tests for backend and ML on every push / PR | `.github/workflows/ci.yml` |

## 2. Setup

- Data: RAVDESS speech, 24 actors (12 M / 12 F), 1,440 utterances; `calm` mapped to `neutral` → 7 classes.
- Single split: 16 / 4 / 4 actors (train / val / test), each val/test split 2 M + 2 F.
- Cross-validation: 6 gender-balanced folds of 4 actors; each actor is tested exactly once.
- Hardware: RTX 3050 Laptop (4 GB) for E1; CPU for E0. Seed 42.

## 3. Results

### 3.1 Headline: speaker-independent 6-fold CV (the number to beat)

| Experiment | WA | UA | Macro-F1 | ECE |
|---|---|---|---|---|
| E0 — thesis MLP (MFCC + pitch + energy) | 0.526 ± 0.070 | 0.516 ± 0.072 | **0.512 ± 0.076** | 0.279 ± 0.069 |
| E1 — repo-style 1D-CNN (MFCC sequence) | 0.557 ± 0.046 | 0.550 ± 0.053 | **0.549 ± 0.050** | 0.200 ± 0.052 |

### 3.2 Speaker leakage inflates accuracy

Same models, same data — only the split differs.

| Experiment | Speaker-independent (single split) | Random utterance split (leaky) | Inflation |
|---|---|---|---|
| E0 macro-F1 | 0.550 | 0.696 | +14.6 pts |
| E1 macro-F1 | 0.538 | 0.750 | +21.2 pts |

The baseline repository's reported ">70%" accuracy is reproduced **only** under the leaky random
split, where the same actors appear in train and test. On unseen speakers, performance is ~51–55%.

### 3.3 Per-class F1 (single speaker-independent split)

| Class | E0 | E1 |
|---|---|---|
| neutral | 0.79 | 0.68 |
| happy | 0.44 | 0.37 |
| sad | 0.38 | 0.43 |
| angry | 0.59 | 0.60 |
| fear | 0.52 | 0.50 |
| disgust | 0.55 | 0.58 |
| surprise | 0.59 | 0.61 |

## 4. Findings

1. **The honest baseline is macro-F1 ≈ 0.51–0.55**, not the 70% claimed by the original repository.
   This is the reference for acceptance criterion AC-01 (TRD §6.3): RAMEI must reach ≥ baseline + 10 pts.
2. **Speaker leakage adds 15–21 points.** All RAMEI results will use speaker-independent splits.
3. **Variance across speakers is large** (±5–8 F1 points between folds). Single-split numbers on
   4 test actors are unreliable; future milestones report cross-validated means.
4. **Baselines are badly calibrated** (ECE 0.20–0.28, target ≤ 0.05). Confidence scores from these
   models cannot be trusted — the core motivation for RAMEI's calibration and reliability modules.
5. **Happy and sad are the hardest classes from audio alone** (F1 0.37–0.44). These are classes
   where words and facial expression add strong evidence — a direct argument for multimodality.
6. **Gender differences are fold-dependent.** CV mean macro-F1 favors female speakers (E1: 0.61 vs 0.48),
   while the single split favors male speakers. With only 24 actors this reflects individual speaker
   variability as much as gender; it must be re-checked on CREMA-D (91 actors) before drawing conclusions (NFR-09).

## 5. Reproduce

```bash
cd ml
python scripts/download_ravdess.py
python -m ramei_ml.data.manifest --dataset ravdess=data/raw/ravdess --out ravdess_audio.csv
python -m ramei_ml.data.manifest --dataset ravdess=data/raw/ravdess --out ravdess_audio_random.csv --strategy random
python -m ramei_ml.training.cv                              # E0, 6-fold
python -m ramei_ml.training.cv experiment=e1_cnn            # E1, 6-fold
python -m ramei_ml.training.baseline data=ravdess_random    # leakage demo
```

## 6. Next: M1 — Unimodal models

| Task | Target |
|---|---|
| Audio branch: WavLM-Base+ with LoRA + attentive pooling (TRD §4.1) | CV macro-F1 ≥ 0.65 on RAVDESS |
| Add CREMA-D (91 actors) for more speaker diversity and a gender check | Cross-corpus numbers |
| Text branch: faster-whisper + DeBERTa-v3 (TRD §4.2) on MELD / GoEmotions | Baseline text F1 |
| Visual branch: face pipeline + ViT FER on RAVDESS video (TRD §4.3) | Baseline visual F1 |
| Temperature scaling on every branch | ECE ≤ 0.05 per branch |
