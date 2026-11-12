import io
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from PIL import Image

class _StubModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.size(0), 10)

@pytest.fixture(autouse=True, scope="session")
def mock_mlflow_model():
    stub = _StubModel().eval()
    with patch("model_loader.load_model", return_value=stub), patch("mlflow.pytorch.load_model", return_value=stub):
        yield stub

