import logging
import os
import mlflow.pytorch
import torch
from torch.export import ExportedProgram

log = logging.getLogger("cifar10_model_loader")

MODEL_NAME = "CifarCNN"
MODEL_ALIAS = "final"

def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    log.info(f"Loading model from MLflow registry: {model_uri}")
    try:
        model = mlflow.pytorch.load_model(model_uri)
        log.info(f"Loaded object type: {type(model)}")

        if isinstance(model, ExportedProgram):
            model = model.module()

        try:
            model.eval()
        except NotImplementedError:
            try:
                from torch.ao.quantization import move_exported_model_to_eval
                move_exported_model_to_eval(model)
            except Exception as e:
                log.warning(f"Could not move exported model to eval mode: {e}")

        log.info("Model loaded successfully.")
        return model
    except Exception as exc:
        log.error(f"Failed to load model from {model_uri}: {exc}")
        raise RuntimeError(f"Could not load model from MLflow registry: {exc}") from exc