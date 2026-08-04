import sys
from pathlib import Path

# Add the 'src' directory to Python's module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pytest

from src.train_cnn import ImageDataset, SimpleCNN

def test_image_dataset_preprocessing():
    """Test data pre-processing: checks shape transformation (HWC -> CHW) and normalization [0, 1]."""
    dummy_X = np.random.randint(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    dummy_y = np.array([0, 1], dtype=np.int64)

    dataset = ImageDataset(dummy_X, dummy_y)
    assert len(dataset) == 2

    img_tensor, label_tensor = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 32, 32)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0
    assert label_tensor.item() == 0


def test_simple_cnn_inference():
    """Test model architecture and forward pass output shape."""
    img_size = 64
    model = SimpleCNN(img_size=img_size)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2)
    assert isinstance(output, torch.Tensor)