import os
import pandas as pd

from evaluation.metrics import evaluate_prediction_file
from evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)
from evaluation.threshold import threshold_tuning


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTION_DIR = os.path.join(BASE_DIR, "results", "predictions")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")
METRIC_DIR = os.path.join(BASE_DIR, "results", "metrics")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def main():
    prediction_files = {}

    for file_name in os.listdir(PREDICTION_DIR):
        if file_name.endswith(".csv"):
            model_name = file_name.replace(".csv", "")
            prediction_files[model_name] = os.path.join(PREDICTION_DIR, file_name)

    if not prediction_files:
        print("No prediction files found.")
        return

    all_results = []

    for model_name, prediction_path in prediction_files.items():
        print(f"Evaluating {model_name}...")

        metrics = evaluate_prediction_file(prediction_path)
        metrics["model"] = model_name

        all_results.append(metrics)

        plot_confusion_matrix(
            prediction_path=prediction_path,
            model_name=model_name,
            output_dir=FIGURE_DIR
        )

        best_threshold = threshold_tuning(
            prediction_path=prediction_path,
            model_name=model_name,
            output_dir=METRIC_DIR
        )

        print(f"Best threshold for {model_name}:")
        print(best_threshold)

    plot_roc_curve(prediction_files, FIGURE_DIR)
    plot_pr_curve(prediction_files, FIGURE_DIR)

    result_df = pd.DataFrame(all_results)

    cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    result_df = result_df[cols]

    output_path = os.path.join(METRIC_DIR, "model_comparison.csv")
    result_df.to_csv(output_path, index=False)

    print("\nModel comparison saved to:")
    print(output_path)

    print("\nFinal comparison:")
    print(result_df)


if __name__ == "__main__":
    main()