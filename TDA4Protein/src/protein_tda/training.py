"""Training and evaluation helpers."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .io import load_npz
from .model import InterfaceMLP, count_parameters


def load_descriptors(descriptor_dir: str | Path, protein_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load all descriptor NPZ files for a split."""

    descriptor_dir = Path(descriptor_dir)
    data = {}
    for protein_id in protein_ids:
        path = descriptor_dir / f"{protein_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing descriptor file: {path}")
        data[protein_id] = load_npz(path)
    return data


def balanced_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Use all positives and a matched number of randomly sampled negatives."""

    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n = min(len(pos), len(neg))
    if n == 0:
        return np.array([], dtype=np.int64)
    chosen = np.concatenate([pos, rng.choice(neg, size=n, replace=False)])
    rng.shuffle(chosen)
    return chosen.astype(np.int64)


def auc_for_split(model: InterfaceMLP, split_data: dict[str, dict[str, np.ndarray]], device: torch.device):
    """Compute per-protein mean/median AUC and all-point AUC."""

    model.eval()
    per_protein = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for arrays in split_data.values():
            labels = arrays["labels"].astype(np.int64)
            if len(np.unique(labels)) < 2:
                continue
            x = torch.as_tensor(arrays["descriptors"], dtype=torch.float32, device=device)
            scores = np.asarray(torch.sigmoid(model(x))[:, 0].detach().cpu().tolist(), dtype=np.float32)
            per_protein.append(roc_auc_score(labels, scores))
            all_labels.append(labels)
            all_scores.append(scores)
    if not per_protein:
        return np.nan, np.nan, np.nan
    all_auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))
    return float(np.mean(per_protein)), float(np.median(per_protein)), float(all_auc)


def train_model(
    descriptor_dir: str | Path,
    train_ids: list[str],
    test_ids: list[str],
    output_dir: str | Path,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (40, 20, 5, 4),
    random_seed: int = 0,
    device_name: str | None = None,
) -> dict[str, float]:
    """Train the interface MLP and write model/history artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)

    train_data = load_descriptors(descriptor_dir, train_ids)
    test_data = load_descriptors(descriptor_dir, test_ids)
    first = next(iter(train_data.values()))
    input_dim = int(first["descriptors"].shape[1])

    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = InterfaceMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    best_test_auc = -np.inf
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tic = time.time()
        model.train()
        losses = []
        for arrays in train_data.values():
            labels = arrays["labels"].astype(np.float32)
            idx = balanced_indices(labels, rng)
            if idx.size == 0:
                continue
            x = torch.as_tensor(arrays["descriptors"][idx], dtype=torch.float32, device=device)
            y_np = np.column_stack([labels[idx], 1.0 - labels[idx]]).astype(np.float32)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_mean, train_median, train_all = auc_for_split(model, train_data, device)
        test_mean, test_median, test_all = auc_for_split(model, test_data, device)
        epoch_record = {
            "epoch": epoch,
            "loss_mean": float(np.mean(losses)) if losses else np.nan,
            "train_auc_mean": train_mean,
            "train_auc_median": train_median,
            "train_auc_all_points": train_all,
            "test_auc_mean": test_mean,
            "test_auc_median": test_median,
            "test_auc_all_points": test_all,
            "epoch_time_s": time.time() - tic,
        }
        history.append(epoch_record)
        if test_mean > best_test_auc:
            best_test_auc = test_mean
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    with (output_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    return {
        "input_dim": input_dim,
        "trainable_parameters": count_parameters(model),
        "epochs": epochs,
        "total_training_time_s": time.time() - start_time,
        "best_test_auc_mean": float(best_test_auc),
        "final_train_auc_mean": history[-1]["train_auc_mean"],
        "final_test_auc_mean": history[-1]["test_auc_mean"],
    }
