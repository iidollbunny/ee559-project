# ee559-project
EE559 Course Project: Credit Card Fraud Detection using Supervised Learning (Logistic Regression, Random Forest, MLP)

This project is the final project for **EE 559: Machine Learning** at the University of Southern California.

The goal of this project is to build and evaluate supervised learning models for credit card fraud detection. The task is formulated as a binary classification problem, where each transaction is classified as either legitimate or fraudulent.

## Project Overview

Credit card fraud detection is an important problem in financial security. Fraudulent transactions can cause financial loss and reduce user trust in financial systems.

A major challenge in this task is the severe class imbalance. Fraudulent transactions account for only about **0.17%** of the dataset. Because of this imbalance, accuracy alone is not a reliable metric. A model can achieve high accuracy by predicting most transactions as legitimate, while still failing to detect fraud cases.

Therefore, this project focuses on evaluation metrics that are more suitable for imbalanced classification, including precision, recall, F1-score, ROC-AUC, and PR-AUC.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains real-world credit card transactions made by European cardholders in September 2013.

The dataset contains:

- 284,807 transactions
- 31 numerical features
- `V1` to `V28`: PCA-transformed anonymized features
- `Time`: transaction time
- `Amount`: transaction amount
- `Class`: target label

The target label is defined as:

- `0`: legitimate transaction
- `1`: fraudulent transaction

The dataset should be placed under:

```text
data/creditcard.csv
Due to file size, the dataset is not included in this repository.

Project Structure
EE559-PROJECT/
├── data/
│   └── creditcard.csv
│
├── results/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── cm_logreg.png
│   │   ├── pr_curve.png
│   │   └── roc_curve.png
│   │
│   ├── metrics/
│   │   ├── baseline.txt
│   │   ├── model_comparison.csv
│   │   ├── threshold_tuning_logreg.csv
│   │   └── threshold_tuning_logreg.png
│   │
│   ├── predictions/
│   │   ├── logreg.csv
│   │   ├── mlp.csv
│   │   ├── rf.csv
│   │   └── rf_smote.csv
│   │
│   └── processed/
│       ├── scaler.pkl
│       ├── split_summary.json
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
│
├── src/
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── threshold.py
│   │
│   ├── train/
│   │   ├── train_mlp.py
│   │   ├── train_rf.py
│   │   └── train_smote.py
│   │
│   ├── eda.py
│   ├── preprocess.py
│   ├── run_evaluation.py
│   └── train_baseline.py
│
├── README.md
└── .gitignore

# Methods

We implemented and compared several supervised learning models.

# Logistic Regression

Logistic Regression is used as the baseline model. Since the dataset is highly imbalanced, class weighting is applied to reduce the bias toward the majority class.

# Random Forest

Random Forest is used to capture non-linear patterns in the dataset. It achieved the best overall balance between precision and recall in our experiments.

# Multi-Layer Perceptron

A Multi-Layer Perceptron is used as a neural network-based model. It improves over the Logistic Regression baseline, but does not outperform Random Forest.

# Random Forest with SMOTE

SMOTE is applied to oversample the minority class and evaluate whether synthetic fraud samples can improve model performance. In our experiments, SMOTE slightly improves PR-AUC, but does not improve the overall F1-score compared with standard Random Forest.

# Preprocessing

The dataset is split into training, validation, and test sets using stratified sampling to preserve the original class distribution.

The split ratio is:

60% training set
20% validation set
20% test set

The Time and Amount features are standardized using StandardScaler. The scaler is fitted only on the training set and then applied to the validation and test sets to avoid data leakage.

Features V1 to V28 are already PCA-transformed and are used directly.

# Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is not sufficient for evaluating model performance. We use the following metrics:

Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion matrix
ROC curve
Precision-recall curve
Threshold tuning analysis

Precision-recall curves are especially important because they better reflect model performance on the rare fraud class.
