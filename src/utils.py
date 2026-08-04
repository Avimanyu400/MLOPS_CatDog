"""Shared helpers: loading processed data, plotting confusion matrix / loss curves."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

PROCESSED_PATH = Path("data/processed/dataset.npz")


def load_processed_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run `python src/data_prep.py` first."
        )
    d = np.load(PROCESSED_PATH, allow_pickle=True)
    classes = list(d["classes"])
    return (
        d["X_train"], d["y_train"],
        d["X_val"], d["y_val"],
        d["X_test"], d["y_test"],
        classes,
    )


def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
