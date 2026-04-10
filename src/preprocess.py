from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = ROOT / "results" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Missing values:")
    print(df.isna().sum().sum())

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    scaler = StandardScaler()
    scale_cols = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    train_df = X_train_scaled.copy()
    train_df["Class"] = y_train.values

    val_df = X_val_scaled.copy()
    val_df["Class"] = y_val.values

    test_df = X_test_scaled.copy()
    test_df["Class"] = y_test.values

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    summary_path = OUTPUT_DIR / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    joblib.dump(scaler, scaler_path)

    summary = {
        "train_shape": train_df.shape,
        "val_shape": val_df.shape,
        "test_shape": test_df.shape,
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean())
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(scaler_path)
    print(summary_path)
    print("Split summary:", summary)

if __name__ == "__main__":
    main()