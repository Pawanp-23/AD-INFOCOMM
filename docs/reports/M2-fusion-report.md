# M2 Report — Reliability + Fusion

| Field | Value |
|---|---|
| Milestone | M2 — Reliability + Fusion (PRD §10) |
| Date | 2026-10-16 |
| Status | Complete — bundle `ramei-rgcmt-1.0.0` exported and verified end-to-end |
| Experiments | E5, E6, E8 (RAMEI), E9, E10 — TRD §6.2 |
| Data | CREMA-D, 7,442 clips, 91 actors, audio + video; 5-fold speaker-independent CV |
| Raw results | [`data/m2_cv_summary.json`](data/m2_cv_summary.json), [`data/m2_analysis_e8.json`](data/m2_analysis_e8.json) |

## 1. Setup

- **Features:** frozen encoders, each clip passed once clean and once per corruption:
  - WavLM-Base+ per-layer statistics (audio: clean, 10 dB and 0 dB SNR);
  - Whisper small.en transcripts, re-run on the noisy audio, encoded by the MELD-trained
    DeBERTa (text: clean ASR, 0 dB ASR, 50% of words dropped);
  - ViT face-expression features on 16 frames (visual: clean, blurred, darkened, occluded).
- **Folds:** 5 gender-balanced folds of speakers. Per fold, one fold is held out for test and
  the next for validation (≈ 18 speakers each); temperature scaling is fitted on validation.
- **Classes:** CREMA-D has **6 emotions** (no surprise); macro-F1 is computed over the classes
  present. Numbers are comparable with each other, but only indicatively with the 7-class
  RAVDESS results of M0/M1.
- **Training:** 25 epochs, AdamW with one-cycle schedule, class weighting, label smoothing 0.1.
  For E8/E9, every example draws per modality the clean version (p = 0.4) or a random
  corruption, and drops a modality with p = 0.2 (at least one is kept).

## 2. Headline

| Model (clean input) | Macro-F1 | ECE |
|---|---|---|
| Audio branch alone | 0.592 ± 0.033 | — |
| Visual branch alone | 0.610 ± 0.015 | — |
| Text branch alone | 0.135 ± 0.005 | — |
| E5 — equal-weight average of branches | 0.721 ± 0.028 | — |
| E6 — reliability-weighted late fusion | 0.717 ± 0.024 | — |
| **E8 — RAMEI (reliability-gated transformer, robust training)** | **0.769 ± 0.023** | **0.044** |

- **+17.7 points over audio-only** and **+15.9 over the best single modality** on the same
  speakers — acceptance criterion NFR-03 (fused ≥ audio baseline + 10 points) is **met**.
- **+4.8 points over simple averaging** (E5): learned fusion is worth having.
- The text branch is near chance on CREMA-D, as expected: every actor reads the same 12
  neutral sentences, so words carry no emotion. The model learned this — mean text
  reliability is 0.21 and text receives ~3% of the fusion weight.

## 3. Robustness (NFR-04: ≤ 10% relative drop)

| Scenario | E8 RAMEI | Drop vs clean | E9 no reliability bias | E10 clean-only training |
|---|---|---|---|---|
| Clean | **0.769** | — | 0.766 | 0.737 |
| Audio noise 10 dB | 0.752 | 2.3% ✅ | 0.756 | 0.583 |
| Noisy room 0 dB (+ ASR on noisy audio) | 0.715 | 7.0% ✅ | 0.719 | 0.384 |
| Face blurred | 0.723 | 5.9% ✅ | 0.733 | 0.587 |
| Face dark | 0.700 | 9.0% ✅ | 0.703 | 0.494 |
| Face half-occluded | 0.699 | 9.1% ✅ | 0.712 | 0.533 |
| 50% of words dropped | 0.769 | 0.0% ✅ | 0.766 | 0.735 |
| Camera off | 0.594 | 22.8% ❌ | 0.609 | 0.534 |
| Microphone off | 0.613 | 20.3% ❌ | 0.615 | 0.560 |
| Noisy room + camera off | 0.464 | 39.6% ❌ | 0.486 | 0.081 |

*Macro-F1, mean of 5 folds (std 0.004-0.034, see raw results).*

- **Every corruption stays within the 10% target.** Removing a whole strong modality does not:
  with the camera off, RAMEI falls back to roughly the audio-only level (0.594 vs 0.592). This is
  the expected ceiling — no fusion can recover information that is absent — and the result
  is graceful degradation, not failure.
- **Robust training is what makes RAMEI robust.** The same architecture trained on clean
  inputs only (E10) collapses under noise (0.384 in a noisy room vs 0.715) and is badly
  calibrated (ECE 0.10-0.37).

## 4. What the reliability module does — and does not — add

**Honest finding: the reliability bias inside the fusion attention gives no measurable
accuracy gain.** E9 (identical training, no reliability bias) matches E8 in every scenario
(differences of 0.00-0.02, inside fold-to-fold noise).

The reliability estimates themselves behave as intended:

| Scenario | Audio reliability | Visual reliability |
|---|---|---|
| Clean | 0.60 | 0.73 |
| Noisy room 0 dB | **0.48** | 0.73 |
| Face dark | 0.60 | **0.63** |
| Face occluded | 0.60 | **0.61** |

But the learned fusion already puts ~84% of its weight on the face for this dataset, and
those weights barely move on average when the voice is degraded. Two reasons:

1. With frozen encoders and corruption-augmented training, the fusion transformer learns to
   discount degraded inputs from the features themselves; the explicit bias is redundant.
2. CREMA-D makes the face the dominant cue (acted, frontal, well-lit video).

What reliability still provides in RAMEI:

- **Explanations** ("Voice down-weighted: background noise", "Face is poorly lit"), driven by
  the same quality signals.
- **Abstention and conflict rules** in the inference engine, which use per-modality
  reliability (bundle thresholds: abstain when reliability < 0.3; conflict needs two
  modalities above 0.6).
- **Per-segment adaptation.** In the end-to-end check, a segment where face reliability fell
  to 0.43 had its weight split 51/49 between face and voice, instead of the usual ~98% face.

ADR-0001 is updated with this outcome. Testing the bias where the face is not dominant
(natural conversations, IEMOCAP or MSP-Podcast) is future work.

## 5. Calibration and abstention (NFR-05, FR-14)

| Scenario | ECE | Abstention rate | Accuracy when committed | Accuracy of abstained |
|---|---|---|---|---|
| Clean | **0.044** ✅ | 4.2% | **78.6%** | 37.3% |
| Noisy room 0 dB | 0.057 | 7.2% | 74.3% | 37.6% |
| Face dark | 0.045 ✅ | 7.5% | 72.7% | 35.4% |
| Camera off | 0.057 | 27.7% | 67.4% | 39.5% |
| Noisy room + camera off | 0.043 ✅ | 45.8% | 58.1% | 35.3% |

*Abstention threshold: calibrated confidence 0.45.*

- Calibration meets the ≤ 0.05 target on clean input and on most corruptions, and is within
  0.007 of it under heavy noise or camera off. It misses on missing-audio (0.087) and
  visual-only (0.100), where a single modality must carry the decision.
- **Abstention does its job.** The segments RAMEI refuses to label are only ~37% accurate,
  versus 79% for the ones it commits to. As conditions worsen it abstains more
  (4% → 28% → 46%) instead of guessing.
- Fitted temperatures are 0.98-1.13: robust training already yields near-calibrated outputs.

## 6. Fairness (NFR-09: gap ≤ 5 F1 points)

| Scenario | Female | Male | Gap |
|---|---|---|---|
| Clean | 0.796 | 0.745 | **5.1** ❌ (just above) |
| Noisy room 0 dB | 0.736 | 0.697 | 3.9 ✅ |
| Face dark | 0.716 | 0.683 | 3.3 ✅ |
| Camera off | 0.596 | 0.593 | 0.2 ✅ |

The gap favours female speakers, the same direction seen on RAVDESS in M0/M1. It is
borderline on clean input and within target elsewhere. It disappears when the face is
removed, which suggests the gap comes from the visual branch (a FER-2013-trained encoder).

## 7. Per-class performance (clean, pooled over folds)

| neutral | happy | sad | angry | fear | disgust |
|---|---|---|---|---|---|
| 0.779 | **0.917** | 0.628 | 0.848 | 0.682 | 0.758 |

The main confusions are sad ↔ fear (157 and 250 clips) and sad ↔ disgust (168 and 110).
These are the low-arousal negative emotions that acted speech and faces express most alike.

## 8. Platform verification

- **Bundle:** `ramei-rgcmt-1.0.0`, trained on all 91 speakers (13 held out for early stopping
  and calibration). Its manifest carries the CV summary shown on the dashboard's System page.
- **End-to-end:** a 17-second video (8 CREMA-D clips: angry, sad, happy) was uploaded through
  the API with consent. Result: 3 segments labelled angry → sad → happy, with confidences
  0.91 / 0.64 / 0.91, transcripts, per-modality reliability and weights, explanations, a
  summary and CSV export. Processing took 4.2 s with models loaded (≈ 4× faster than real
  time on the RTX 3050), and 38 s on a cold start including model loading.
  *This checks the pipeline, not accuracy — the bundle has seen these speakers.*
- **Restart safety:** sessions interrupted by a server restart are now recovered on startup
  (found and fixed during this check).

## 9. Reproduce

```bash
cd ml
python scripts/run_m2.py                                   # features -> CV -> bundle (resumable)
python -m ramei_ml.fusion.analysis artifacts/experiments/m2_fusion_crema_d/<run>
```

Feature extraction took about 6 hours on the RTX 3050, dominated by Whisper ASR on two audio
variants. Fusion CV (3 experiments × 5 folds) took 21 minutes, and bundle export 2 minutes.
