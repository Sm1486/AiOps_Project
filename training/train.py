import json
import logging
import os
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from AiOps_Project.training.data_pipeline import build_dataloaders

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CifarCNN-train")

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
MODEL_PATH = ROOT / "models" / "CifarCNN.pt"
METRICS_PATH = ROOT / "metrics" / "train_metrics.json"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_NAME = "CifarCNN"
MODEL_NAME = "CifarCNN"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

DEFAULT_PARAMS = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam", "val_split": 0.1, "seed": 42, "num_ray_workers": 4, "ray_shard_size": 5000}

def load_params():
    params = dict(DEFAULT_PARAMS)
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            params.update(yaml.safe_load(f) or {})
    return params

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
 
    return running_loss / total, correct / total

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
 
    return running_loss / total, correct / total, np.concatenate(all_preds), np.concatenate(all_labels)

def plot_training_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True", title="Confusion Matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def move_to_production(run_id):
    client = MlflowClient()
    versions = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
    if not versions:
        log.warning("Could not find the registered model version for this run.")
        return
    version = str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(MODEL_NAME, "final", version)
    log.info(f"Model '{MODEL_NAME}' v{version} moved to Production")

def train():
    params = load_params()
    torch.manual_seed(int(params["seed"]))
    np.random.seed(int(params["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.set_tags({"model_type": "CifarCNN", "dataset": "CIFAR-10", "data_loader": "ray", "device": str(device)})
        train_loader, val_loader, test_loader, pipeline_stats = build_dataloaders(params, device)
        mlflow.log_metrics(pipeline_stats)

        model = CifarCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        lr = float(params["learning_rate"])
        if str(params["optimizer"]).lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc, best_state = 0.0, None

        for epoch in range(int(params["epochs"])):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = val_epoch(model, val_loader, criterion, device)
            epoch_time = time.time() - t0
            for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[k].append(v)
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc, "epoch_time_sec": epoch_time}, step=epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            log.info(f"Epoch {epoch + 1}/{params['epochs']} | "
                     f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                     f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.1f}s")

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc, preds, targets = val_epoch(model, test_loader, criterion, device)
        log.info(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.4f}")
        cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        np.add.at(cm, (targets, preds), 1)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc, "best_val_accuracy": best_val_acc})
        mlflow.log_metrics({f"test_acc_{c}": float(a) for c, a in zip(CLASSES, class_acc)})
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = REPORT_DIR / "confusion_matrix.png"
        curves_path = REPORT_DIR / "training_curves.png"
        plot_confusion_matrix(cm, cm_path)
        plot_training_curves(history, curves_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_dict({c: float(a) for c, a in zip(CLASSES, class_acc)}, "class_accuracy.json")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "best_val_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "final_train_accuracy": history["train_acc"][-1],
                "final_train_loss": history["train_loss"][-1],
                "data_pipeline_samples_per_sec": pipeline_stats["data_pipeline_samples_per_sec"],
            }, f, indent=2)

        model = model.cpu().eval()
        input_example = np.random.randn(1, 3, 32, 32).astype(np.float32)
        with torch.no_grad():
            output_example = model(torch.from_numpy(input_example)).numpy()
        signature = infer_signature(input_example, output_example)
        mlflow.pytorch.log_model(model, artifact_path="model",signature=signature,input_example=input_example,registered_model_name=MODEL_NAME)
        move_to_production(run.info.run_id)
        log.info(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()