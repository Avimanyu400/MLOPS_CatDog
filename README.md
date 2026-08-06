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

## Project structure

```
MLOPS_CatDog/
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

```
python -m venv .venv && source .venv/lib/activate
pip install -r requirements.txt
pip install --upgrade pip
```

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

# Verify the confusion matrix images, loss curve plots
# Reports are stored under reports/figures/
 

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

The DVC remote is currently set to a local path (`dvc remote list`). 

## Git workflow

```bash
git log --oneline
```
Code, `dvc.yaml`/`dvc.lock`, `*.dvc` pointer files, and `params.yaml` are versioned
in Git. Large binary data (`data/raw`, `data/processed`) and trained models are
tracked by DVC / left out of Git per `.gitignore` — only their lightweight `.dvc`
pointer files live in Git, keeping the repo small while data stays reproducible.

#  Start Uvicorn API gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# To test /predict endpoint
python .\src\test_api.py

# To test /health endpoint
http://localhost:8000/health

# Environment specification

pip freeze > requirements.txt

# Docker container build
docker build -t cat-dog-classification-api:v1

# Run Docker
docker run -p 8080:8080 cat-dog-classification-api:v1

# Start Minikunbe
minikube start  --driver=docker

# check minikube status
minikube status

kubectl cluster-info
kubectl get node
minikube docker-env

# Unit testing using Pytest
pytest

# Github action
create file .github/workflows/ci-cd.yml

# Push the changes to github to trigger github action workflows
git add src/train_cnn.py
git commit -m "Fix cross-platform log path in train_cnn.py for CI/CD runners"
git push origin main

# Now verify the github action in github web browser. Both workflows should be completed successfully.
