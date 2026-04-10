import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create output directory for figures
os.makedirs('results/figures', exist_ok=True)

# Load dataset
df = pd.read_csv('data/creditcard.csv')

# Print basic dataset information
print("Dataset shape:", df.shape)

print("\nClass distribution (counts):")
print(df['Class'].value_counts())

print("\nClass distribution (ratio):")
print(df['Class'].value_counts(normalize=True))

# Plot class distribution (important for imbalanced dataset)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)

plt.title('Class Distribution (0 = Normal, 1 = Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('results/figures/class_distribution.png')
plt.show()

# Plot transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Amount'], bins=50)

plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/figures/amount_distribution.png')
plt.show()