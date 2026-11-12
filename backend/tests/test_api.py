import pytest
import torch
import AiOps_Project.backend.inference_app as inference_app
from fastapi.testclient import TestClient
from AiOps_Project.backend.inference_app import app
import io
from PIL import Image

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 10)

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    monkeypatch.setattr(inference_app, "model", DummyModel())

client = TestClient(app)

def make_test_image():
    img = Image.new("RGB", (32, 32), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck']

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

class TestPredictEndpoint:
    def test_predict_valid_image_returns_200(self):
        response = client.post("/predict", files={"file": make_test_image()})
        assert response.status_code == 200

    def test_predict_returns_valid_class(self):
        response = client.post("/predict", files={"file": make_test_image()})
        result = response.json()
        assert result["predicted_class"] in CIFAR10_CLASSES

    def test_predict_returns_confidence(self):
        response = client.post("/predict", files={"file": make_test_image()})
        assert 0.0 <= response.json()["confidence"] <= 1.0

    def test_predict_missing_file_returns_422(self):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_latency_under_2_seconds(self):
        import time
        start = time.time()
        client.post("/predict", files={"file": make_test_image()})
        assert time.time() - start < 2.0