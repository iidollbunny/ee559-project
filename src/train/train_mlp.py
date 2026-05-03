import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)

TRAIN_PATH = "results/processed/train.csv"
VAL_PATH = "results/processed/val.csv"
TEST_PATH = "results/processed/test.csv"
TARGET = "Class"


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return df, X, y


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_on_val(model, X_val, y_val):
    val_prob = model.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_prob)
    best_threshold, best_f1 = find_best_threshold(y_val, val_prob)
    return val_prob, val_ap, best_threshold, best_f1


def main():
    train_df, X_train, y_train = load_data(TRAIN_PATH)
    val_df, X_val, y_val = load_data(VAL_PATH)
    test_df, X_test, y_test = load_data(TEST_PATH)

    configs = [
        {
            "hidden_layer_sizes": (64,),
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
        },
        {
            "hidden_layer_sizes": (128, 64),
            "alpha": 1e-4,
            "learning_rate_init": 5e-4,
        },
    ]

    best_model = None
    best_config = None
    best_val_f1 = -1.0
    best_val_ap = -1.0
    best_threshold = 0.5

    for i, cfg in enumerate(configs, start=1):
        print(f"\n[CONFIG {i}] {cfg}")

        model = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            batch_size=2048,
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
            verbose=False,
        )

        model.fit(X_train, y_train)

        val_prob, val_ap, threshold, val_f1 = evaluate_on_val(model, X_val, y_val)
        print(f"[VAL] AUPRC: {val_ap:.4f}")
        print(f"[VAL] Best threshold: {threshold:.2f}, Best F1: {val_f1:.4f}")

        if (val_f1 > best_val_f1) or (val_f1 == best_val_f1 and val_ap > best_val_ap):
            best_model = model
            best_config = cfg
            best_val_f1 = val_f1
            best_val_ap = val_ap
            best_threshold = threshold

    print("\n========== BEST CONFIG ==========")
    print(best_config)
    print(f"[VAL] Best AUPRC: {best_val_ap:.4f}")
    print(f"[VAL] Best threshold: {best_threshold:.2f}")
    print(f"[VAL] Best F1: {best_val_f1:.4f}")

    val_prob = best_model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= best_threshold).astype(int)

    print("\n[VAL] Classification report:")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    test_ap = average_precision_score(y_test, test_prob)

    print(f"\n[TEST] AUPRC: {test_ap:.4f}")
    print("[TEST] Classification report:")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    out = pd.DataFrame(
        {
            "id": np.arange(len(test_df)),
            "y_true": y_test.values,
            "y_pred": test_pred,
            "y_prob": test_prob,
        }
    )
    out.to_csv("results/predictions/mlp.csv", index=False)
    print("\nSaved to results/predictions/mlp.csv")


if __name__ == "__main__":
    main()