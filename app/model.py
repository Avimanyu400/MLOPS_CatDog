"""
Model loading and inference utilities for the serving layer.

Kept independent of the training script's dependencies (mlflow, sklearn
training code, etc.) so the inference container stays lean. The SimpleCNN
architecture is duplicated here intentionally -- it must match
`train_cnn.py::SimpleCNN` exactly since we load its state_dict.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/cnn_baseline.pt")


class SimpleCNN(nn.Module):
    """Must stay identical to the architecture used at training time."""

    def __init__(self, img_size: int = 64):
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


def preprocess_image(image_bytes: bytes, img_size: int) -> torch.Tensor:
    """Decode raw image bytes into a normalized (1, 3, H, W) float tensor.

    This is the pure, testable preprocessing function: resize -> RGB ->
    scale pixel values to [0, 1] -> HWC to CHW -> add batch dim.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0, 1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)
    return tensor


class ModelBundle:
    """Loads a checkpoint once and exposes a simple predict() call."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model: SimpleCNN | None = None
        self.img_size: int = 64
        self.classes: List[str] = ["cat", "dog"]

    def load(self) -> "ModelBundle":
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{path}'. "
                "Mount/copy the trained .pt file or set MODEL_PATH."
            )
        checkpoint = torch.load(path, map_location="cpu")
        self.img_size = checkpoint.get("img_size", 64)
        self.classes = checkpoint.get("classes", ["cat", "dog"])
        model = SimpleCNN(img_size=self.img_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        return self

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> Tuple[str, List[float]]:
        """Run inference on raw image bytes, returning (label, probabilities)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        tensor = preprocess_image(image_bytes, self.img_size)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()
        pred_idx = int(np.argmax(probs))
        label = self.classes[pred_idx]
        return label, probs
