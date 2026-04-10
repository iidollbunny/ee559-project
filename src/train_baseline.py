import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Set paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
results_dir = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2. Load data
# =========================
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# =========================
# 3. Split features and label
# =========================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain/Test split completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")
print(f"y_test distribution:\n{y_test.value_counts()}")

# =========================
# 4. Scale the Amount feature
# =========================
X_train = X_train.copy()
X_test = X_test.copy()

scaler = StandardScaler()
X_train.loc[:, "Amount"] = scaler.fit_transform(X_train[["Amount"]]).ravel()
X_test.loc[:, "Amount"] = scaler.transform(X_test[["Amount"]]).ravel()

print("\nFeature scaling completed for 'Amount'.")

# =========================
# 5. Train Logistic Regression baseline
# =========================
print("\nTraining Logistic Regression baseline model...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")

# =========================
# 6. Make predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. Evaluate model
# =========================
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

# =========================
# 8. Save results to file
# =========================
output_path = os.path.join(results_dir, "baseline.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("Logistic Regression Baseline Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"X_train shape: {X_train.shape}\n")
    f.write(f"X_test shape: {X_test.shape}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

print(f"\nResults saved to: {output_path}")
print("\nBaseline pipeline completed successfully.")