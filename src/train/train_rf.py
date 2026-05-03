import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score, f1_score

TRAIN_PATH = "results/processed/train.csv"
VAL_PATH   = "results/processed/val.csv"
TEST_PATH  = "results/processed/test.csv"

TARGET = "Class"

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return df, X, y

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds 的长度比 precisions/recalls 少 1
    best_t = 0.5
    best_f1 = -1

    for t in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def main():
    train_df, X_train, y_train = load_data(TRAIN_PATH)
    val_df, X_val, y_val = load_data(VAL_PATH)
    test_df, X_test, y_test = load_data(TEST_PATH)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # validation
    val_prob = model.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_prob)
    best_t, best_f1 = find_best_threshold(y_val, val_prob)

    print(f"[VAL] AUPRC: {val_ap:.4f}")
    print(f"[VAL] Best threshold: {best_t:.2f}, Best F1: {best_f1:.4f}")

    val_pred = (val_prob >= best_t).astype(int)
    print("\n[VAL] Classification report:")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    # test
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_t).astype(int)
    test_ap = average_precision_score(y_test, test_prob)

    print(f"\n[TEST] AUPRC: {test_ap:.4f}")
    print("[TEST] Classification report:")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    # save predictions
    out = pd.DataFrame({
        "id": np.arange(len(test_df)),
        "y_true": y_test.values,
        "y_pred": test_pred,
        "y_prob": test_prob
    })
    out.to_csv("results/predictions/rf.csv", index=False)
    print("\nSaved to results/predictions/rf.csv")

if __name__ == "__main__":
    main()