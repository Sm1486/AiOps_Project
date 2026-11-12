import io
import os
import uvicorn
import logging
import time
import torch
from torchvision import transforms
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Summary, Counter, Gauge, Histogram, Info, start_http_server, disable_created_metrics
from AiOps_Project.backend.model_loader import load_model

log = logging.getLogger("cifar10_backend")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

disable_created_metrics()

REQUEST_COUNT = Counter('api_request_total', 'Total API requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Inference latency in seconds', ['endpoint'], buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
IMAGE_SIZE_BYTES = Summary('image_size_bytes', 'Size of uploaded images in bytes')
CONFIDENCE_GAUGE = Gauge('last_prediction_confidence', 'Confidence of the last prediction')

build_info = Info('cifar10_build', 'CIFAR-10 Classifier Build Info')
build_info.info({'version': '1.0.0', 'model': 'CifarCNN', 'dataset': 'CIFAR-10'})

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    start_http_server(18000)
    yield
    log.info("Shutting down CIFAR-10 backend")
    model = None

app = FastAPI(title="CIFAR-10 Classifier API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    try:
        contents = await file.read()
        image_size = len(contents)
        IMAGE_SIZE_BYTES.observe(image_size)
        img = (Image.open(io.BytesIO(contents)).convert("RGB").resize((32, 32)))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        predicted_class = CLASSES[probs.argmax().item()]
        confidence = float(probs.max())
        all_probs = {cls: float(p) for cls, p in zip(CLASSES, probs)}
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        CONFIDENCE_GAUGE.set(confidence)

        log.info(f"Predicted: {predicted_class} " f"({confidence:.2%}) in {latency:.3f}s")
        return {"predicted_class": predicted_class, "confidence": confidence, "all_probabilities": all_probs, "latency_ms": round(latency * 1000, 2)}

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        log.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)