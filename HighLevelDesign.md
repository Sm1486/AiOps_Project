#  High-Level Design (HLD) — CIFAR-10 Image Classifier


## System Architecture Overview

```
Docker Compose

Frontend (Streamlit) ----->  Backend (FastAPI)
Port : 8501          <-----  Port : 8000
                                    |
                                    |
                             Mlflow Model Registry(CifarCNN Model)
                             Port : 5000
                                    |
                                    |
Grafana              <-----  Prometheus Exporter
Port : 3000                  Port : 18000
```

The frontend and backend are **loose-coupled**. They communicate only through the configurable REST API calls. The backend URL is given to the application via an environment variable (`BACKEND_URL`) so both components can be deployed and scaled independently.

---

## 3. Component Design Choices & Rationale
- Model: Custom CNN - CifarCNN
- Backend: FastAPI
- Frontend: Streamlit
- Data Pipeline: Ray
- Experiment Tracking: MLflow
- CI Pipeline: DVC
-  Monitoring: Prometheus + Grafana
- Packaging: Docker + Docker Compose
---

## 4. Data Flow

1. **Training time**: `dvc repro` → Ray pipeline downloads and preprocesses CIFAR-10 → `train.py` trains CifarCNN → MLflow logs metrics & registers model → `evaluate.py` writes `eval_metrics.json`
2. **Inference time**: User uploads image via Streamlit → HTTP POST to FastAPI `/predict` → FastAPI loads image, runs CNN inference → returns JSON `{predicted_class, confidence, all_probabilities, latency_ms}` → Streamlit renders result
3. **Monitoring**: Every `/predict` call increments Prometheus counters and histograms → Prometheus scrapes every 5s → Grafana displays live dashboard

---
