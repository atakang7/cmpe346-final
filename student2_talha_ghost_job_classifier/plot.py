"""
Generate result figures for the ghost-job classification experiment.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


RESULTS_DIR = "results"
CLASS_NAMES = ["Realistic", "Likely Ghost"]


def plot_loss_curves(loss_histories):
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {"LSTM": "o", "BERT": "s", "RoBERTa": "^"}
    colors = {"LSTM": "#e15759", "BERT": "#4e79a7", "RoBERTa": "#59a14f"}
    for name, losses in loss_histories.items():
        epochs = list(range(1, len(losses) + 1))
        ax.plot(epochs, losses, marker=markers[name], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss per Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison_bar(results):
    models = list(results.keys())
    acc = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4e79a7")
    ax.bar(x + width / 2, f1, width, label="Weighted F1", color="#59a14f")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Ghost-Job Classification Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrices(all_preds):
    models = list(all_preds.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        y_true, y_pred = all_preds[name]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
