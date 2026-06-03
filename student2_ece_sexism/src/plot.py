import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_path="../results/results.csv", output_path="../results/comparison.png"):
    """
    Reads the model results from a CSV file and generates a grouped bar chart
    comparing Accuracy and F1 scores across models.

    Args:
        results_path : path to the results.csv file
        output_path  : path where the generated chart will be saved
    """
    df = pd.read_csv(results_path)

    models = df["model"].tolist()
    accuracy = df["accuracy"].tolist()
    f1 = df["f1"].tolist()

    x = np.arange(len(models))  # model konumları
    width = 0.35  # çubuk genişliği

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1, width, label="F1 (Macro)")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Sexism Detection (EDOS)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Çubukların üstüne değerleri yaz
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    plot_results()