# ml/

Research and training for RAMEI: datasets, unimodal branches, reliability module, adaptive
fusion, evaluation and ONNX export. Spec: [TRD §3–6](../docs/02-TRD.md#3-data-requirements) ·
Architecture: [SAD §6](../docs/03-System-Architecture.md#6-ml-architecture).

```
ml/
├── configs/                 # Hydra configs
│   ├── data/                #   dataset + split definitions
│   ├── model/               #   branch / reliability / fusion settings
│   └── experiment/          #   E0–E11 experiment presets
├── data/
│   ├── manifests/           # versioned split CSVs (path, speaker_id, split, label) — tracked
│   ├── raw/                 # downloaded datasets — git-ignored
│   └── processed/           # cached segments, transcripts, face crops — git-ignored
├── notebooks/               # exploration only; nothing production depends on notebooks
├── src/ramei_ml/
│   ├── data/                # dataset classes, label mapping, corruption augmentations
│   ├── features/            # MFCC baseline features, quality features
│   ├── models/              # audio/text/visual branches, reliability, fusion, heads
│   ├── training/            # stages S0 (baseline) → S4 (calibration)
│   ├── evaluation/          # metrics, robustness, calibration, fairness reports
│   └── export/              # ONNX export + model bundle builder
├── scripts/                 # download / preprocessing entry points
├── artifacts/               # local outputs and bundles — git-ignored
└── tests/
```

## Setup

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate                     # Windows (source .venv/bin/activate elsewhere)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -e ".[dev]"
pytest                                     # runs on synthetic audio, no dataset needed
```

## M0 — reproduce the baselines

```bash
python scripts/download_ravdess.py                                         # ~208 MB
python -m ramei_ml.data.manifest --dataset ravdess=data/raw/ravdess --out ravdess_audio.csv
python -m ramei_ml.data.manifest --dataset ravdess=data/raw/ravdess --out ravdess_audio_random.csv --strategy random

python -m ramei_ml.training.baseline                                       # E0 thesis MLP
python -m ramei_ml.training.baseline experiment=e1_cnn                     # E1 repo 1D-CNN
python -m ramei_ml.training.baseline data=ravdess_random                   # leakage demo
```

Results: `artifacts/experiments/leaderboard.csv`, plus per-run `metrics.json`, confusion matrix,
predictions and model under `artifacts/experiments/<experiment>__<data>/<timestamp>/`.

## Training stages

| Stage | Output |
|---|---|
| S0 | Thesis/repo baseline (MFCC + MLP / 1D-CNN), speaker-independent split |
| S1 | Audio (WavLM), Text (Whisper + DeBERTa), Visual (ViT FER) branches |
| S2 | Reliability Estimation Module |
| S3 | Reliability-Gated Cross-Modal Transformer fusion |
| S4 | Calibration and abstention thresholds |

## Hand-off to the backend

Training ends by building a **model bundle** (format in
[SAD §6.4](../docs/03-System-Architecture.md#64-model-bundle-contract-between-ml-and-backend)).
The backend only ever consumes bundles — never code or checkpoints from this folder.

## Data rules

- Use only datasets listed in TRD §3.1, under their licenses.
- Splits are **speaker-independent**; manifests are the single source of truth.
- Never commit audio, video, or model weights.
