"""
Baseline model #2: A simple Convolutional Neural Network (PyTorch).

Logs params/metrics/loss-curves/confusion-matrix to MLflow and saves the
trained model as a .pt (state_dict) file.

Usage:
    python src/train_cnn.py --epochs 10 --batch-size 32 --lr 1e-3
"""
import argparse
import time
from pathlib import Path
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from datetime import datetime

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from utils import load_processed_data, plot_confusion_matrix, plot_loss_curves
import logging
import os

# Define your log file path
log_file_path = r"D:\MLOPS_CatDog\logs\cnn_model_training.log"

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


class ImageDataset(Dataset):
    """Wraps uint8 HWC image arrays as normalized CHW float tensors."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0  # HWC, [0,1]
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img), torch.tensor(self.y[idx], dtype=torch.long)


class SimpleCNN(nn.Module):
    """A small 3-block CNN: Conv-BN-ReLU-Pool x3 -> FC -> 2 classes."""

    def __init__(self, img_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        reduced = img_size // 8
        self.fc1 = nn.Linear(64 * reduced * reduced, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return avg_loss, preds, labels


def main():

    parser = argparse.ArgumentParser(description="Train a CNN for Cat-Dog Classification")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--experiment", type=str, default="dog-cat-classification", help="Experiment name")

    # 2. Parse the arguments
    args = parser.parse_args()

    # 3. Assign them to local variables
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EXPERIMENT_NAME = args.experiment
    logger.info(f"Running experiment '{EXPERIMENT_NAME}' with epochs={EPOCHS}, batch_size={BATCH_SIZE} and learning_rate={LEARNING_RATE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_processed_data()
    img_size = X_train.shape[1]

    train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ImageDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=args.batch_size)

    # Generate a clean timestamp string (e.g., 2026-07-31_22-45-12)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine base name with the timestamp using an f-string
    run_name = f"cnn_baseline_{timestamp}"

    with mlflow.start_run(run_name=run_name):
    #with mlflow.start_run(run_name="cnn_baseline"):
        mlflow.log_params({
            "model_type": "SimpleCNN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "img_size": img_size,
            "device": str(device),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        model = SimpleCNN(img_size=img_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses, val_losses = [], []
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            val_acc = accuracy_score(val_labels, val_preds)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)
            #print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
            logger.info(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", train_time)

        # Loss curve artifact
        loss_curve_path = "reports/figures/cnn_loss_curves.png"
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        mlflow.log_artifact(loss_curve_path)
        logger.info(f"Loss Curve saved to {loss_curve_path}")

        # Test evaluation
        _, test_preds, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        test_metrics = {
            "test_accuracy": accuracy_score(test_labels, test_preds),
            "test_precision": precision_score(test_labels, test_preds),
            "test_recall": recall_score(test_labels, test_preds),
            "test_f1": f1_score(test_labels, test_preds),
        }
        mlflow.log_metrics(test_metrics)

        # Confusion matrix artifact
        cm_path = "reports/figures/cnn_confusion_matrix.png"
        plot_confusion_matrix(test_labels, test_preds, classes, cm_path)
        mlflow.log_artifact(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Save model as .pt and log as artifact
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "cnn_baseline.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "img_size": img_size,
            "classes": classes,
        }, model_path)
        mlflow.log_artifact(str(model_path))

        # 1. Define the input and output schemas with dynamic batch dimension (-1)
        input_schema = Schema(
             [TensorSpec(np.dtype(np.float32), (-1, 3, img_size, img_size), name="input")]
            )
        output_schema = Schema(
                [TensorSpec(np.dtype(np.float32), (-1, len(classes)), name="output")]
                )

        # 2. Create the Model Signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # 3. Create a numpy-compatible example input
        example_input = np.zeros((1, 3, img_size, img_size), dtype=np.float32)

        # 4. Log the model using serialization_format="pickle"
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            input_example=example_input,
            signature=signature,
            serialization_format="pickle",  # Bypasses pt2 strict tracing constraints
            )

        metrics_str = ", ".join([f"{key}: {value}" for key, value in test_metrics.items()])
        logger.info(f"Test Metrics -> {metrics_str}")
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
