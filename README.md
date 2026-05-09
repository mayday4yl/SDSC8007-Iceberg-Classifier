# SDSC8007 Iceberg Classifier

This repository contains the reproducible code for our SDSC8007 mini project on the Kaggle **Statoil/C-CORE Iceberg Classifier Challenge**.

The task is binary classification: given two SAR image bands and the radar incidence angle, predict the probability that a target is an iceberg rather than a ship.

## Final Result

The competition still supports late submission, so the final project score is reported using the official Kaggle binary log loss.

```text
Final submission file:      submission_blend_8models_20260503_212025.csv
Kaggle private score:       0.08098
Kaggle public score:        0.09987
Local strict 5-fold CV:     0.088556
Top-5-level reference:      0.08883
```

The local CV score is used as a reproducibility check. It is computed from out-of-fold predictions using the same binary log loss formula as Kaggle.

## Repository Layout

```text
README.md
requirements.txt

data/processed/                 Place Kaggle raw files here
docs/final_report.md            Final report source

src/iceberg/data.py             Data loading, channel construction, folds, pseudo-label filtering
src/iceberg/models.py           CNN, VGG-style CNN, and FiLM ResNet34 models
src/iceberg/metrics.py          Binary log loss

scripts/build_cache.py          Convert Kaggle JSON files into NumPy cache
scripts/train_cnn.py            Train first-stage CNN models
scripts/angle_stack.py          Strict incidence-angle second-stage stacking
scripts/blend_predictions.py    Blend second-stage predictions
scripts/verify_final_oof.py     Recompute final local CV log loss
scripts/run_from_scratch.sh     One-command reproduction script
```

## Data

The Kaggle data files are not included in this repository. After downloading and extracting the competition data, place exactly these files under `data/processed/`:

```text
data/processed/train.json
data/processed/test.json
data/processed/sample_submission.csv
```

## Environment

The final run was reproduced on:

```text
GPU:     NVIDIA GeForce RTX 4090 24GB
CUDA:    12.8
Python:  3.12
PyTorch: 2.6.0a0+ecf3bae40a.nv25.01
Seed:    2026
```

Install the Python dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

This repository does not pin a PyTorch wheel because GPU servers usually provide a CUDA-matched PyTorch build. Before running, verify that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Reproduce From Scratch

After placing the data files under `data/processed/`, run:

```bash
PYTHON_BIN=python bash scripts/run_from_scratch.sh
```

The script trains the first-stage CNN models, builds strict incidence-angle stacking models, blends the second-stage predictions, and writes all outputs to:

```text
artifacts/from_scratch_<timestamp>/
  run.log
  predictions/
  reports/
  models/
```

Key generated files:

```text
predictions/submission_blend_8models_*.csv
predictions/oof_blend_8models_*.csv
reports/metrics_blend_8models_*.json
run.log
```

Upload `predictions/submission_blend_8models_*.csv` to Kaggle late submission to obtain the official score.

## Method Summary

The final system has two stages:

1. Train several SAR image CNN models with 5-fold cross-validation.
2. Combine their out-of-fold predictions with incidence-angle statistical features using LightGBM and Logistic Regression stacking.

The final blend contains 8 second-stage models: 4 LightGBM stackers and 4 Logistic Regression stackers trained with different random seeds. The blend weights are optimized on out-of-fold binary log loss.

To reduce leakage risk, label-based incidence-angle statistics for a validation fold are computed without using that fold's true labels.
