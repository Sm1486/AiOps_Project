# Low-Level Design (LLD) 

## Service Configuration

| Property | Value |
|----------|-------|
| Framework | FastAPI 0.111+ |
| Host | `0.0.0.0` |
| Port | `8000` |
| Prometheus metrics port | `18000` |
| Model source | MLflow Model Registry — `models:/CifarCNN@final` |
| Logging format | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` |


## API Endpoints

### 1. `GET /health`

**Purpose**: Confirms the server is running and the model has been loaded.

**Request**

```
GET /health HTTP/1.1
Host: localhost:8000
```

**Response — 200 OK**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` when the server is reachable |
| `model_loaded` | `boolean` | `true` if the MLflow model was loaded successfully at startup |

**Error responses**: None expected.

---

### 2 `POST /predict`

**Purpose**: Classify an uploaded image

**Request**

```
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----FormBoundary

------FormBoundary
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

<binary image data>
------FormBoundary--
```

| Field | Location | Type | Required | Description |
|-------|----------|------|----------|-------------|
| `file` | multipart body | `image/jpeg` or `image/png` | Yes | The image to classify.|

**Response — 200 OK**

```json
{
  "predicted_class": "cat",
  "confidence": 0.8731,
  "all_probabilities": {
    "airplane":    0.0012,
    "automobile":  0.0008,
    "bird":        0.0154,
    "cat":         0.8731,
    "deer":        0.0021,
    "dog":         0.0844,
    "frog":        0.0006,
    "horse":       0.0093,
    "ship":        0.0018,
    "truck":       0.0113
  },
  "latency_ms": 42.7
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | `string` | The class with the highest probability |
| `confidence` | `float` | Softmax probability of the predicted class|
| `all_probabilities` | `object` | Softmax probabilities for all 10 classes|
| `latency_ms` | `float` | End-to-end inference time from the server (in ms)|

**Error responses**

| HTTP Code | Condition | Body |
|-----------|-----------|------|
| `422 Unprocessable Entity` | `file` field is missing from the request | FastAPI default validation error JSON |
| `500 Internal Server Error` | Model inference raised an exception | FastAPI default error JSON and error is logged |
---

## Prometheus Metrics Specification

All metrics are exposed at `http://backend:18000/metrics`.

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `api_request_total` | Counter | `endpoint`, `status` | Total API requests partitioned by the endpoint path and outcome (`success`/`error`) |
| `inference_latency_seconds` | Histogram | `endpoint` | Server-side inference latency in buckets: 0.01, 0.05, 0.1, 0.5, 1.0, 5.0 seconds |
| `image_size_bytes` | Summary | — | Size of each uploaded image in bytes |
| `last_prediction_confidence` | Gauge | — | Softmax confidence of the most recent prediction |
| `cifar10_build_info` | Info | `version`, `model`, `dataset` | Static build metadata |

---

## Model Loader Module

### `load_model() → torch.nn.Module`

| Aspect | Detail |
|--------|--------|
| Input | None (reads `MLFLOW_TRACKING_URI` env var, defaults to `http://localhost:5000`) |
| Output | `torch.nn.Module` in eval mode |
| On error | Raises `RuntimeError` with a descriptive message |
| Model URI | `models:/CifarCNN@final` |

---

## Software Design Paradigm

The backend follows **Object-Oriented + Functional** Python style:
- The FastAPI `app` is a module-level singleton which is an OO pattern for lifecycle management.
- Pure functions are used for preprocessing, metric recording which are stateless operations.
- There is a proper separation of concerns, `model_loader.py` for model loading and `inference_app.py` for the HTTP layer.