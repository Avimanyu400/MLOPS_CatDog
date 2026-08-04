import io

import torch
from PIL import Image

from app.model import ModelBundle, SimpleCNN


def _make_image_bytes(img_size=64) -> bytes:
    img = Image.new("RGB", (img_size, img_size), color=(10, 200, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_simplecnn_forward_output_shape():
    model = SimpleCNN(img_size=64)
    model.eval()
    dummy_input = torch.zeros((2, 3, 64, 64))
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 2)  # batch of 2, 2 classes


def test_model_bundle_predict_returns_valid_label_and_probs(tmp_path):
    # Build and save a tiny untrained checkpoint so ModelBundle.load() has
    # something real to read, without depending on a trained artifact.
    img_size = 64
    classes = ["cat", "dog"]
    model = SimpleCNN(img_size=img_size)
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "img_size": img_size, "classes": classes},
        checkpoint_path,
    )

    bundle = ModelBundle(model_path=str(checkpoint_path)).load()
    assert bundle.is_loaded

    label, probs = bundle.predict(_make_image_bytes(img_size))

    assert label in classes
    assert len(probs) == len(classes)
    assert abs(sum(probs) - 1.0) < 1e-4  # softmax outputs sum to 1
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_model_bundle_predict_without_load_raises():
    bundle = ModelBundle(model_path="does_not_matter.pt")
    try:
        bundle.predict(_make_image_bytes())
        assert False, "Expected RuntimeError when predicting before load()"
    except RuntimeError:
        pass


def test_model_bundle_load_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.pt"
    bundle = ModelBundle(model_path=str(missing_path))
    try:
        bundle.load()
        assert False, "Expected FileNotFoundError for missing checkpoint"
    except FileNotFoundError:
        pass
