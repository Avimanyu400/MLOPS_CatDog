import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.model import preprocess_image


def _make_image_bytes(size=(100, 50), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_image_output_shape():
    img_size = 64
    image_bytes = _make_image_bytes()
    tensor = preprocess_image(image_bytes, img_size)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, img_size, img_size)


def test_preprocess_image_normalizes_to_unit_range():
    image_bytes = _make_image_bytes(color=(255, 255, 255))
    tensor = preprocess_image(image_bytes, img_size=32)

    assert tensor.max().item() <= 1.0
    assert tensor.min().item() >= 0.0
    # A pure white image should normalize to (close to) all ones.
    assert torch.allclose(tensor, torch.ones_like(tensor), atol=1e-6)


def test_preprocess_image_handles_non_rgb_input():
    # Grayscale ("L" mode) input must still convert cleanly to 3-channel RGB.
    img = Image.new("L", (40, 40), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    tensor = preprocess_image(buf.getvalue(), img_size=32)
    assert tensor.shape == (1, 3, 32, 32)


def test_preprocess_image_rejects_garbage_bytes():
    with pytest.raises(Exception):
        preprocess_image(b"not an image", img_size=32)
