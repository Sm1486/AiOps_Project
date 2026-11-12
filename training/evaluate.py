import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("CifarCNN-evaluate")

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
MODEL_PATH = ROOT / "models" / "CifarCNN.pt"
EVAL_METRICS_PATH = ROOT / "metrics" / "eval_metrics.json"
REPORT_DIR = ROOT / "reports"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DEFAULT_PARAMS = {"batch_size": 128, "seed": 42}

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

    def forward(self, x):
        return self.classifier(self.features(x))

def load_params():
    params = dict(DEFAULT_PARAMS)
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            params.update(yaml.safe_load(f) or {})
    return params

def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
        xticklabels=CLASSES, yticklabels=CLASSES,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix — Evaluation Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"Confusion matrix saved to {path}")

def evaluate():
    params = load_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. " "Run the 'train' DVC stage first.")
    model = CifarCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    log.info(f"Loaded model weights from {MODEL_PATH}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_ds = CIFAR10(root=str(ROOT / "data" / "cifar10"), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=int(params["batch_size"]), shuffle=False)
    log.info(f"Test set size: {len(test_ds)} samples")
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    elapsed = time.time() - t0
    test_loss = running_loss / total
    test_acc = correct / total
    throughput = total / elapsed
    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (all_labels_np, all_preds_np), 1)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    log.info(f"Evaluation complete: loss={test_loss:.4f}, accuracy={test_acc:.4f}, " f"throughput={throughput:.1f} samples/sec, elapsed={elapsed:.2f}s")
    for cls, acc in zip(CLASSES, class_acc):
        log.info(f"  {cls:>12s}: {acc:.4f}")

    EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 6),
        "eval_throughput_samples_per_sec": round(throughput, 2),
        "eval_elapsed_sec": round(elapsed, 3),
        **{f"test_acc_{cls}": round(float(acc), 6) for cls, acc in zip(CLASSES, class_acc)},
    }
    with open(EVAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Evaluation metrics written to {EVAL_METRICS_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, REPORT_DIR / "eval_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
