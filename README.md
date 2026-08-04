# Dog vs Cat Classification — Baseline Pipeline

Baseline image-classification pipeline for the Kaggle
[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset).

Implements:
- **Two baseline models**: Logistic Regression on flattened pixels, and a simple CNN (PyTorch).
- **Model serialization**: `.pkl` (sklearn) and `.pt` (PyTorch state_dict).
- **Experiment tracking**: MLflow — logs params, metrics, confusion matrices, and loss curves.
- **Source control**: Git for code.
- **Data versioning**: DVC for `data/raw` (source images) and a DVC pipeline stage for
  `data/processed` (preprocessed numpy arrays).

> **Note on this repo as delivered**: `data/raw/{Cat,Dog}` currently contains small
> synthetic placeholder images (generated in-repo) so the full pipeline could be
> validated end-to-end without network access to Kaggle. Swap in the real dataset
> (same folder layout) before training for real — no code changes needed.

## Project structure

```
dog-cat-classifier/
├── data/
│   ├── raw/              # Cat/, Dog/ — DVC-tracked source images
│   └── processed/        # dataset.npz — DVC-pipeline output (preprocessed arrays)
├── src/
│   ├── data_prep.py      # resize, split, save to .npz
│   ├── train_logreg.py   # baseline 1: logistic regression on flattened pixels
│   ├── train_cnn.py      # baseline 2: simple CNN
│   └── utils.py          # shared plotting / data-loading helpers
├── models/                # saved .pkl / .pt models (gitignored; see below)
├── reports/figures/       # confusion matrices, loss curves (MLflow artifacts)
├── dvc.yaml / dvc.lock    # DVC pipeline definition (data_prep stage)
├── params.yaml            # pipeline hyperparameters tracked by DVC
├── requirements.txt
└── mlflow.db / mlruns/    # MLflow tracking store (gitignored)
```

## Setup environment

```bash
python -m venv .venv && source .venv/lib/activate
pip install -r requirements.txt
pip install --upgrade pip
```

## 1. Get the real data

Download the Kaggle dataset (requires a free Kaggle account + API token):

```bash
pip install kaggle

# Download dataset from kaggle and store under below paths:
data/raw/Cat/ 
data/raw/Dog/ 
```
 

## 2. Preprocess (via DVC pipeline)

```bash
dvc repro
```

This resizes images, splits train/val/test, and writes `data/processed/dataset.npz`.
Re-running `dvc repro` after changing `params.yaml` or the raw data will only
re-execute the stage if its dependencies actually changed.

## 3. Train baselines

```bash
# Baseline 1: Logistic Regression
python src/train_logreg.py --C 1.0 --max-iter 200

# Baseline 2: Simple CNN
python src/train_cnn.py --epochs 10 --batch-size 32 --lr 1e-3
```

Each run logs to MLflow (SQLite backend, local file `mlflow.db`) and saves a model
to `models/`.

## 4. View experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000 to compare runs, params, metrics, and artifacts
(confusion matrix images, loss curve plots, serialized models).

## 5. Data versioning with DVC

```bash
# See what's tracked
cat data/raw.dvc
cat dvc.yaml

# Push data to remote storage (configured as a local folder here;
# swap for S3/GCS/Azure in production — see `dvc remote modify`)
dvc push

# On another machine / after `git clone`:
dvc pull
```

The DVC remote is currently set to a local path (`dvc remote list`). For a real
team setup, repoint it at shared storage, e.g.:
```bash
dvc remote modify local_storage url s3://your-bucket/dvc-storage
```

## Git workflow

```bash
git log --oneline
```
Code, `dvc.yaml`/`dvc.lock`, `*.dvc` pointer files, and `params.yaml` are versioned
in Git. Large binary data (`data/raw`, `data/processed`) and trained models are
tracked by DVC / left out of Git per `.gitignore` — only their lightweight `.dvc`
pointer files live in Git, keeping the repo small while data stays reproducible.

## Results (synthetic placeholder data — for pipeline validation only)

| Model | Test Accuracy | Test F1 |
|---|---|---|
| Logistic Regression | 1.00 | 1.00 |
| Simple CNN | 1.00 | 1.00 |

These are trivially perfect because the placeholder images have an easy synthetic
color signal baked in — they only prove the pipeline runs end-to-end. Expect the
CNN to meaningfully outperform logistic regression on the real photographic dataset.
