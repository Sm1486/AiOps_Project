import logging
import time
from pathlib import Path
import numpy as np
import ray
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

log = logging.getLogger("cifar10_data")

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
DATA_DIR = ROOT / "data" / "cifar10"

MEAN, STD = 0.5, 0.5

@ray.remote
def preprocess_shard(images, labels, start, end):
    x = images[start:end].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return x, labels[start:end].astype(np.int64)


def load_split_with_ray(train, shard_size):
    ds = CIFAR10(root=str(DATA_DIR), train=train, download=True)
    n = len(ds.data)

    images_ref = ray.put(ds.data)
    labels_ref = ray.put(np.asarray(ds.targets, dtype=np.int64))

    futures = [
        preprocess_shard.remote(images_ref, labels_ref, s, min(s + shard_size, n))
        for s in range(0, n, shard_size)
    ]
    shards = ray.get(futures)

    x = torch.from_numpy(np.concatenate([s[0] for s in shards]))
    y = torch.from_numpy(np.concatenate([s[1] for s in shards]))
    return x, y


def load_cifar10_with_ray(num_workers = 4, shard_size = 5000):
    ray.init(
        num_cpus=int(num_workers),
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level=logging.WARNING,
        object_store_memory=1_500_000_000,
        _temp_dir="/var/tmp/ray",
    )
    t0 = time.time()
    x_train, y_train = load_split_with_ray(train=True, shard_size=shard_size)
    x_test, y_test = load_split_with_ray(train=False, shard_size=shard_size)
    elapsed = time.time() - t0
    ray.shutdown()

    total = len(y_train) + len(y_test)
    stats = {"data_pipeline_seconds": elapsed, "data_pipeline_samples_per_sec": total / elapsed}
    log.info(f"Ray pipeline: {total} samples in {elapsed:.2f}s " f"-> {stats['data_pipeline_samples_per_sec']:.0f} samples/sec")
    return x_train, y_train, x_test, y_test, stats

class TensorImageDataset(Dataset):
    def __init__(self, x, y, pad = 4):
        self.x, self.y, self.pad = x, y, pad

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]
        return img, self.y[idx]


def build_dataloaders(params, device):
    x_train, y_train, x_test, y_test, stats = load_cifar10_with_ray(num_workers=params["num_ray_workers"], shard_size=params["ray_shard_size"])

    n = len(y_train)
    gen = torch.Generator().manual_seed(int(params["seed"]))
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * float(params["val_split"]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(TensorImageDataset(x_train, y_train), train_idx)
    val_ds = Subset(TensorImageDataset(x_train, y_train), val_idx)
    test_ds = TensorImageDataset(x_test, y_test)

    batch_size = int(params["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    log.info(f"Split sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, stats

def run_pipeline():
    num_workers, shard_size = 4, 5000
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            p = yaml.safe_load(f) or {}
        num_workers = p.get("num_ray_workers", num_workers)
        shard_size = p.get("ray_shard_size", shard_size)

    _, _, _, _, stats = load_cifar10_with_ray(num_workers, shard_size)
    print(f"[THROUGHPUT] {stats['data_pipeline_samples_per_sec']:.0f} samples/sec " f"({stats['data_pipeline_seconds']:.2f}s total)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_pipeline()