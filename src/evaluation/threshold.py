import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score


def threshold_tuning(prediction_path: str, model_name: str, output_dir: str):
    df = pd.read_csv(prediction_path)

    y_true = df["y_true"]
    y_prob = df["y_prob"]

    thresholds = [i / 100 for i in range(1, 100)]

    records = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        records.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    result_df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"threshold_tuning_{model_name}.csv")
    result_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(result_df["threshold"], result_df["precision"], label="Precision")
    plt.plot(result_df["threshold"], result_df["recall"], label="Recall")
    plt.plot(result_df["threshold"], result_df["f1"], label="F1-score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Tuning - {model_name}")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"threshold_tuning_{model_name}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    best_row = result_df.loc[result_df["f1"].idxmax()]

    return best_row