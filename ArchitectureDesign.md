#  Architecture Design

```
Docker Compose

Frontend (Streamlit) ----->  Backend (FastAPI)
Port : 8501          <-----  Port : 8000
                                    |
                                    |
                             Mlflow Model Registry (CifarCNN Model)
                             Port : 5000
                                    |
                                    |
Grafana              <-----  Prometheus Exporter
Port : 3000                  Port : 18000
```

---

## Frontend
* Uses a streamlit application to display the model.
* You can upload an image and it'll send a request to the backend to classify it.
* Checks whether the image fits the required format as well(png or jpg).

## Backend
* Predicts the class of the image using the trained model from the MLflow Model Registry

## Mlflow Model Registry
* Loads and processes the Cifar-10 data.
* Trains the model and stores the model and metrics.
* Serves the model and can be used by the backend.

## Prometheus Exporter
* Saves all the metrics we're checking for.

## Grafana
* Gives us a dashboard for all our metrics.
