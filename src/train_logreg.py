"""
Baseline model #1: Logistic Regression on flattened, normalized pixel values.

Logs params/metrics/artifacts to MLflow and saves the trained model as a .pkl.

Usage:
    python src/train_logreg.py --C 1.0 --max-iter 200
"""
import argparse
import pickle
import time
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import load_processed_data, plot_confusion_matrix
from datetime import datetime
import logging

import os

# Define your log file path
log_file_path = r"D:\MLOPS_CatDog\logs\logreg_model_training.log"

# 1. Extract the directory path from the full file path
log_dir = os.path.dirname(log_file_path)

# 2. Check and create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# 3. Open (or create) the log file for writing
with open(log_file_path, "a") as f:
    f.write("Training started successfully.\n")

print(f"Log path verified/created at: {log_file_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, mode="a"), 
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def flatten_and_normalize(X):
    return X.reshape(len(X), -1).astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--experiment", type=str, default="dog-cat-classification")
    args = parser.parse_args()

    C_value = args.C
    max_iterations = args.max_iter
    experiment_name = args.experiment
    logger.info(f"Running experiment '{experiment_name}' with C={C_value} and max_iter={max_iterations}")
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_processed_data()

    X_train_f = flatten_and_normalize(X_train)
    X_val_f = flatten_and_normalize(X_val)
    X_test_f = flatten_and_normalize(X_test)

    # Generate a clean timestamp string (e.g., 2026-07-31_22-45-12)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine base name with the timestamp using an f-string
    run_name = f"Logistic_reg_baseline_{timestamp}"

    with mlflow.start_run(run_name=run_name):
    #with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": args.C,
            "max_iter": args.max_iter,
            "img_size": X_train.shape[1],
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        t0 = time.time()
        clf = LogisticRegression(C=args.C, max_iter=args.max_iter)
        clf.fit(X_train_f, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", train_time)

        # Validation metrics
        val_pred = clf.predict(X_val_f)
        mlflow.log_metrics({
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_precision": precision_score(y_val, val_pred),
            "val_recall": recall_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred),
        })

        # Test metrics
        test_pred = clf.predict(X_test_f)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_precision": precision_score(y_test, test_pred),
            "test_recall": recall_score(y_test, test_pred),
            "test_f1": f1_score(y_test, test_pred),
        }
        mlflow.log_metrics(test_metrics)

        # Confusion matrix artifact
        cm_path = "reports/figures/logreg_confusion_matrix.png"
        plot_confusion_matrix(y_test, test_pred, classes, cm_path)
        mlflow.log_artifact(cm_path)
        logger.info(f"Confusion Matrix saved to {cm_path}")

        # Save model as .pkl and log as artifact
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "logreg_baseline.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")

        #print("Test metrics:", test_metrics)
        #print(f"Model saved to {model_path}")
        metrics_str = ", ".join([f"{key}: {value}" for key, value in test_metrics.items()])
        logger.info(f"Test Metrics -> {metrics_str}")
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
