# CIFAR-10 Image Classifier — AIOps Assignment

A production-grade CIFAR-10 image classification system built with AIOps principles.

## AI Usage Disclaimer
* Used it to know where to add the logging code so all the important sections get logged.
* Used it for the generalising the Paths in the python codes
* For test class generation - I gave the specifics, but the exact code was generated


## Running Training

* Run "pip install -r requirements.txt"
* Run "mlflow server --host 0.0.0.0 --port 5000"
* Open another terminal and run the following commands to initialize DVC
- git lfs install
- dvc init
* Run "docker compose up -d mlflow"
* Run "export MLFLOW_TRACKING_URI=http://localhost:5000"
* Run "dvc repro --force" to run the full pipeline (data loading, training, evaluation)
* Run "dvc dag" 
* "docker compose up --build"
* To run the tests, run the following commands
- cd backend
- touch conftest.py
- pytest
* (We need to run touch conftest.py to deal with the dependencies that happen with the import statements from other files in the directory)

## Quick Start

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000/docs |
| MLflow tracking | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus metrics exporter | http://localhost:18000/metrics |