# ml/

Research and training for RAMEI: datasets, unimodal branches, reliability module, adaptive
fusion, evaluation and ONNX export. Spec: [TRD §3–6](../docs/02-TRD.md#3-data-requirements) ·
Architecture: [SAD §6](../docs/03-System-Architecture.md#6-ml-architecture).

```
ml/
├── configs/                 # Hydra configs
│   ├── data/                #   dataset + split definitions (speaker-independent by default)
│   └── experiment/          #   E0-E2 experiment presets
├── data/
│   ├── manifests/           # versioned split CSVs (path, speaker_id, split, label) - tracked
│   ├── raw/                 # downloaded datasets - git-ignored
│   └── processed/           # 16 kHz audio, face crops, fusion feature store - git-ignored
├── src/ramei_ml/
│   ├── data/                # dataset parsers, label mapping, manifests, corruptions
│   ├── features/            # MFCC, audio loading, ASR, face video, quality features
│   ├── models/              # baseline MLP/CNN, audio (WavLM+LoRA) and text (DeBERTa) branches
│   ├── training/            # baseline, audio and text training, speaker cross-validation
│   ├── fusion/              # M2: feature extraction, reliability-gated fusion, CV, bundle export
│   └── evaluation/          # metrics, calibration (ECE, temperature scaling)
├── scripts/                 # dataset downloads, run_m2.py pipeline driver, CV summaries
├── artifacts/               # experiments, models and bundles - git-ignored
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

## M1 — unimodal branches

```bash
pip install -e ".[audio,multimodal]"
python scripts/download_crema_d.py                                # CREMA-D audio
python -m ramei_ml.training.audio_branch                          # E2: WavLM + LoRA, 6-fold speaker CV
python -m ramei_ml.training.audio_branch experiment=e2_audio_frozen
python -m ramei_ml.training.text_branch                           # DeBERTa-v3 + LoRA on MELD
python scripts/download_crema_d_video.py                          # face crops for the visual branch
python scripts/summarize_cv.py artifacts/experiments/e2_audio_wavlm_lora__ravdess_cv6
```

## M2 — reliability-gated fusion and the model bundle

```bash
python scripts/run_m2.py      # resumable: index -> audio/ASR/text/visual features -> fusion CV -> bundle
```

Steps it runs (each can be run alone): `python -m ramei_ml.fusion.extract {index,audio,asr,text,visual}`,
`python -m ramei_ml.fusion.train` (5-fold speaker CV over clean/noisy/missing-modality scenarios),
`python -m ramei_ml.fusion.export --version 1.0.0 --cv-results <results.json>`.
Output: `artifacts/bundles/ramei-rgcmt-1.0.0/` (manifest, weights, calibration, CV metrics).

| Stage | Output |
|---|---|
| S0 | Thesis/repo baseline (MFCC + MLP / 1D-CNN), speaker-independent split |
| S1 | Audio (WavLM), Text (Whisper + DeBERTa), Visual (ViT FER) branches |
| S2 | Per-modality reliability from quality features |
| S3 | Reliability-gated fusion over frozen encoder features |
| S4 | Temperature calibration and abstention threshold |

## Hand-off to the backend

Training ends by building a **model bundle** (format in
[SAD §6.4](../docs/03-System-Architecture.md#64-model-bundle-contract-between-ml-and-backend)).
The backend only ever consumes bundles — never code or checkpoints from this folder.

## Data rules

- Use only datasets listed in TRD §3.1, under their licenses.
- Splits are **speaker-independent**; manifests are the single source of truth.
- Never commit audio, video, or model weights.
