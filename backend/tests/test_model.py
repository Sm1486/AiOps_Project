import io
import pytest
import torch
import sys
import os
import mlflow.pytorch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from AiOps_Project.backend.model_loader import load_model

class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

@pytest.fixture(scope="module")
def model():
    m = CifarCNN(num_classes=10)
    m.eval()
    return m

@pytest.fixture
def random_image_batch():
    return torch.randn(4, 3, 32, 32)

@pytest.fixture
def single_image():
    return torch.randn(1, 3, 32, 32)

class TestModelArchitecture:
    def test_output_shape_batch(self, model: CifarCNN, random_image_batch: torch.Tensor):
        with torch.no_grad():
            out = model(random_image_batch)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_output_shape_single(self, model: CifarCNN, single_image: torch.Tensor):
        with torch.no_grad():
            out = model(single_image)
        assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"

    def test_forward_pass_no_exception(self, model: CifarCNN, single_image: torch.Tensor):
        try:
            with torch.no_grad():
                model(single_image)
        except Exception as exc:
            pytest.fail(f"Forward pass raised an exception: {exc}")

    def test_softmax_probabilities_sum_to_one(self, model: CifarCNN, single_image: torch.Tensor):
        with torch.no_grad():
            logits = model(single_image)
            probs = torch.softmax(logits, dim=1)[0]
        total = float(probs.sum())
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected ~1.0"

    def test_all_classes_have_probability(self, model: CifarCNN, single_image: torch.Tensor):
        with torch.no_grad():
            logits = model(single_image)
            probs = torch.softmax(logits, dim=1)[0]
        assert len(probs) == 10
        assert all(float(p) >= 0.0 for p in probs)

    def test_model_in_eval_mode(self, model: CifarCNN):
        assert not model.training, "Model should be in eval mode"

    def test_model_parameters_exist(self, model: CifarCNN):
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"

class TestModelLoader:
    def test_load_model_raises_on_bad_uri(self, monkeypatch):
        def boom(uri):
            raise Exception("registry unreachable")
        monkeypatch.setattr(mlflow.pytorch, "load_model", boom)
        with pytest.raises(RuntimeError):
            load_model()