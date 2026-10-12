# M1 Report — Unimodal Branches

| Field | Value |
|---|---|
| Milestone | M1 — Unimodal models (PRD §10) |
| Date | 2026-10-13 |
| Status | Complete |
| Experiments | E2 / E2a audio (TRD §4.1), text branch (TRD §4.2), visual pipeline (TRD §4.3) |

## 1. Audio branch — WavLM-Base+ (RAVDESS, 6-fold speaker-independent CV)

| Experiment | Encoder | Macro-F1 | WA | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| E0 thesis MLP (M0) | MFCC + pitch + energy | 0.512 ± 0.076 | 0.526 | 0.279 | — |
| E1 repo 1D-CNN (M0) | MFCC sequence | 0.549 ± 0.050 | 0.557 | 0.200 | — |
| **E2 WavLM + LoRA** (r=8) | fine-tuned attention | **0.686 ± 0.028** | 0.699 | 0.083 | 0.089 |
| **E2a WavLM frozen** | layer weights + pooling + head | **0.701 ± 0.062** | 0.711 | 0.088 | 0.079 |

- **+17 to +19 macro-F1 points over the thesis baseline** on unseen speakers, clearing the M1 target
  (≥ 0.65) and the PRD's "baseline + 10" acceptance criterion for the audio modality alone.
- **LoRA vs frozen is a statistical tie** on 24 actors: frozen has the higher mean but twice the
  fold-to-fold spread. With so few speakers, fine-tuning the encoder does not pay off — which is why
  the M2 fusion uses frozen WavLM features (ADR-0008).
- **Calibration needs more validation data.** Temperature scaling fitted on 4 validation speakers
  per fold did not transfer reliably (E2: 0.083 → 0.089). M2 calibrates on ~18 speakers per fold.
- **Run-to-run variance is real.** Repeating E2 with seed 43 on folds 0–2 gave 0.712 / 0.685 / 0.527
  vs 0.645 / 0.695 / 0.692 for seed 42 (the seed-43 run stopped after fold 2 and is excluded from the
  table). Model selection on 4 validation speakers is noisy; early stopping (patience 4) also stopped
  fold 0 at epoch 6.
- Gender: female 0.714 / male 0.644 (E2) — same direction as M0, re-examined on CREMA-D in M2.

## 2. Text branch — DeBERTa-v3-base + LoRA (MELD, official splits)

| Metric (test, 2,610 utterances) | Value |
|---|---|
| Macro-F1 | **0.448** |
| Weighted-F1 | **0.591** |
| Weighted accuracy | 0.560 |
| ECE raw → calibrated (T = 0.84) | 0.122 → **0.082** |

Per-class F1: neutral 0.69 · happy 0.62 · surprise 0.57 · angry 0.46 · sad 0.42 · disgust 0.20 · fear 0.18.

- Weighted-F1 0.59 is in line with published text-only results on MELD; fear and disgust are rare
  (≈2% each) and remain hard.
- Trained on natural conversational text so the branch is useful on real calls. On acted corpora
  (RAVDESS, CREMA-D) every actor reads the same few sentences, so text carries almost no emotion
  signal there — a deliberate test for the reliability module in M2.

## 3. Visual pipeline

- MediaPipe BlazeFace detection → square crop with 25% margin → 224×224, **16 frames per clip**,
  with per-frame detection confidence, face area, blur (Laplacian variance) and brightness.
- CREMA-D: **7,442 / 7,442 clips** processed, faces found in essentially every frame
  (example: mean detection confidence 0.95).
- Encoder: `trpakov/vit-face-expression` (ViT-B/16 trained on FER-2013, same 7 classes) kept frozen;
  a temporal transformer head is trained inside the fusion model. Branch-only results are reported
  in the M2 report (scenario "visual only").
- RAVDESS video was planned but Zenodo throttled downloads to ~8 KB/s (13 GB); CREMA-D video
  (91 actors, 2.1 GB) was used instead — 4× more speakers for fusion evaluation.

## 4. Engineering notes

| Issue | Fix |
|---|---|
| WavLM LayerDrop removed hidden states during training → layer-weighted sum crashed | LayerDrop disabled; regression test added |
| transformers v5 loads checkpoints in stored dtype (DeBERTa fp16) → fp16 gradients | Encoders loaded explicitly in fp32 |
| CREMA-D Git-LFS batch transfer stalled | Sparse clone for LFS pointers + parallel HTTPS with sha256 verification |
| Session end interrupted face extraction (10 half-written clips) | Integrity scan + resumable re-fetch |

## 5. Reproduce

```bash
cd ml
python -m ramei_ml.training.audio_branch                          # E2, 6 folds
python -m ramei_ml.training.audio_branch experiment=e2_audio_frozen
python -m ramei_ml.training.text_branch                           # MELD
python scripts/download_crema_d_video.py                          # face crops
python scripts/summarize_cv.py artifacts/experiments/e2_audio_wavlm_lora__ravdess_cv6
```
