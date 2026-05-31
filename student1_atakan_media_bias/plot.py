"""
Generates the 3 report figures and saves them to results/.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = "results"
CLASS_NAMES = ["Not Biased", "Biased"]


def plot_loss_curves(loss_histories):
    """Line plot of training loss per epoch for all models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {"LSTM": "o", "BERT": "s", "RoBERTa": "^"}
    colors = {"LSTM": "#e15759", "BERT": "#4e79a7", "RoBERTa": "#59a14f"}
    for name, losses in loss_histories.items():
        epochs = list(range(1, len(losses) + 1))
        ax.plot(epochs, losses, marker=markers[name], color=colors[name],
                linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss per Epoch", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_bar(results):
    """Grouped bar chart comparing Accuracy and F1 across models."""
    models = list(results.keys())
    acc = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4e79a7")
    bars2 = ax.bar(x + width / 2, f1, width, label="F1 Score", color="#59a14f")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Accuracy vs F1", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrices(all_preds):
    """One confusion matrix per model in a single figure."""
    models = list(all_preds.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        y_true, y_pred = all_preds[name]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
