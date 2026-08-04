"""Plotting helpers for manuscript and SI figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_training_auc(history_csv: str | Path, output_png: str | Path) -> None:
    """Plot mean and median train/test ROC AUC curves."""

    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(df["epoch"], df["train_auc_mean"], label="Train mean", linewidth=2)
    ax.plot(df["epoch"], df["train_auc_median"], label="Train median", linewidth=2)
    ax.plot(df["epoch"], df["test_auc_mean"], label="Test mean", linewidth=2)
    ax.plot(df["epoch"], df["test_auc_median"], label="Test median", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def pca_projection_and_variance(
    descriptors: np.ndarray,
    labels: np.ndarray,
    output_prefix: str | Path,
    sample_per_class: int = 50000,
    random_seed: int = 0,
    variance_threshold: float = 0.95,
) -> None:
    """Create a two-panel PCA projection and cumulative-variance figure plus CSVs."""

    rng = np.random.default_rng(random_seed)
    scaler = StandardScaler()
    x = scaler.fit_transform(descriptors)
    pca = PCA(random_state=random_seed).fit(x)
    projection = pca.transform(x)[:, :2]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    threshold_component = int(np.searchsorted(cumulative, variance_threshold) + 1)

    chosen = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        if idx.size > sample_per_class:
            idx = rng.choice(idx, size=sample_per_class, replace=False)
        chosen.append(idx)
    chosen = np.concatenate(chosen)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pc1": projection[chosen, 0],
            "pc2": projection[chosen, 1],
            "iface": labels[chosen],
        }
    ).to_csv(output_prefix.with_name(output_prefix.name + "_projection_sample.csv"), index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": cumulative,
        }
    ).to_csv(output_prefix.with_name(output_prefix.name + "_variance.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    colors = np.where(labels[chosen] == 1, "#C44E52", "#4C72B0")
    axes[0].scatter(projection[chosen, 0], projection[chosen, 1], c=colors, s=2, alpha=0.25, linewidths=0)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].set_title("(a) PCA projection")
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=2, color="#333333")
    axes[1].axhline(variance_threshold, color="#C44E52", linestyle="--", linewidth=1.5)
    axes[1].axvline(threshold_component, color="#C44E52", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Number of principal components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("(b) PCA variance")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".svg"))
    plt.close(fig)
