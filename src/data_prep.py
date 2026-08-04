"""
Data preparation for the Dog vs Cat classifier.

Reads raw images from data/raw/{Cat,Dog}/*.jpg (the structure produced when you
unzip the Kaggle "Dog and Cat Classification Dataset"), resizes them, splits
into train/val/test, and writes preprocessed numpy arrays to data/processed/.

This processed output is what gets versioned with DVC (see dvc.yaml / README),
so preprocessing is reproducible and doesn't need to be re-run by every user.

Usage:
    dvc repro
    OR
    python src/data_prep.py --img-size 64 --val-split 0.15 --test-split 0.15
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CLASSES = ["Cat", "Dog"]  # label 0 = Cat, label 1 = Dog


def load_image_paths():
    paths, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Expected raw images at {cls_dir}. Download the Kaggle dataset "
                f"and place images under data/raw/Cat and data/raw/Dog."
            )
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(f)
                labels.append(label_idx)
    return paths, labels


def load_and_resize(path, img_size):
    try:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        print(f"Skipping unreadable file {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    paths, labels = load_image_paths()
    print(f"Found {len(paths)} raw images across {len(CLASSES)} classes.")

    data = []
    valid_labels = []
    for p, l in zip(paths, labels):
        arr = load_and_resize(p, args.img_size)
        if arr is not None:
            data.append(arr)
            valid_labels.append(l)

    X = np.stack(data)  # (N, H, W, 3) uint8
    y = np.array(valid_labels, dtype=np.int64)

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    n = len(X)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DIR / "dataset.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        img_size=args.img_size,
        classes=np.array(CLASSES),
    )

    print(f"Saved processed dataset to {PROCESSED_DIR / 'dataset.npz'}")
    print(f"  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")


if __name__ == "__main__":
    main()
