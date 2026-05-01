import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)


def plot_confusion_matrix(prediction_path: str, model_name: str, output_dir: str):
    df = pd.read_csv(prediction_path)

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )

    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(prediction_files: dict, output_dir: str):
    plt.figure(figsize=(6, 5))

    for model_name, prediction_path in prediction_files.items():
        df = pd.read_csv(prediction_path)

        y_true = df["y_true"]
        y_prob = df["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(prediction_files: dict, output_dir: str):
    plt.figure(figsize=(6, 5))

    for model_name, prediction_path in prediction_files.items():
        df = pd.read_csv(prediction_path)

        y_true = df["y_true"]
        y_prob = df["y_prob"]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=model_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()