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
```


## Methods

We implemented and compared several supervised learning models for the fraud detection task.

### Logistic Regression

Logistic Regression is used as the baseline model. Since the dataset is highly imbalanced, class weighting is applied to reduce the bias toward the majority class.

### Random Forest

Random Forest is used to capture non-linear patterns in the dataset. It achieved the best overall balance between precision and recall in our experiments.

### Multi-Layer Perceptron

A Multi-Layer Perceptron is used as a neural network-based model. It improves over the Logistic Regression baseline, but does not outperform Random Forest.

### Random Forest with SMOTE

SMOTE is applied to oversample the minority class and evaluate whether synthetic fraud samples can improve model performance. In our experiments, SMOTE slightly improves PR-AUC, but does not improve the overall F1-score compared with standard Random Forest.

## Preprocessing

The dataset is split into training, validation, and test sets using stratified sampling to preserve the original class distribution.

### Data Split

- Training set: 60%
- Validation set: 20%
- Test set: 20%

### Feature Scaling

The `Time` and `Amount` features are standardized using `StandardScaler`. The scaler is fitted only on the training set and then applied to the validation and test sets to avoid data leakage.

Features `V1` to `V28` are already PCA-transformed and are used directly.

## Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is not sufficient for evaluating model performance. We use metrics that better reflect fraud detection performance.

### Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- ROC curve
- Precision-recall curve
- Threshold tuning analysis

Precision-recall curves are especially important because they better reflect model performance on the rare fraud class.

## How to Run

This project can be run step by step from data preprocessing to model evaluation. Before running the scripts, make sure the dataset is saved as `data/creditcard.csv`.

### Install Dependencies

Install the required Python packages using `pip install -r requirements.txt`.

If the SMOTE-based model is used, install the additional package with `pip install imbalanced-learn`.

### Exploratory Data Analysis

Run `python src/eda.py` to perform exploratory data analysis. This step generates basic dataset information and the class distribution figure.

### Data Preprocessing

Run `python src/preprocess.py` to split and preprocess the dataset.

This step creates the training, validation, and test sets under `results/processed/`. The `Time` and `Amount` features are standardized, while the PCA-transformed features `V1` to `V28` are used directly.

### Baseline Model Training

Run `python src/train_baseline.py` to train the Logistic Regression baseline model.

This script saves the baseline results to `results/metrics/baseline.txt` and generates the prediction file `results/predictions/logreg.csv`.

### Improved Model Training

Run the improved model scripts to train Random Forest, MLP, and Random Forest with SMOTE.

The corresponding scripts are `src/train/train_rf.py`, `src/train/train_mlp.py`, and `src/train/train_smote.py`.

After training, each model should generate a prediction file under `results/predictions/`. The expected prediction files are `logreg.csv`, `rf.csv`, `mlp.csv`, and `rf_smote.csv`.

### Model Evaluation

Run `python src/run_evaluation.py` after all prediction files are generated.

This evaluation script reads all prediction files from `results/predictions/` and produces the final comparison results, including `results/metrics/model_comparison.csv`, `results/figures/roc_curve.png`, and `results/figures/pr_curve.png`.

## Prediction File Format

Each model should output a prediction CSV file with four columns: `id`, `y_true`, `y_pred`, and `y_prob`.

The `id` column represents the sample index. The `y_true` column stores the ground-truth label, while `y_pred` stores the predicted class label. The `y_prob` column stores the predicted probability of the fraud class.

All prediction files should be saved under `results/predictions/`. The evaluation pipeline automatically reads all CSV files in this directory.

## Results Summary

The Logistic Regression baseline achieves high fraud recall but very low precision. This means it can detect most fraud cases, but it also produces many false positives.

Random Forest achieves the best overall performance. It provides a strong balance between precision and recall and obtains the highest F1-score among the tested models.

The MLP model also improves over the baseline, but it performs slightly worse than Random Forest.

Random Forest with SMOTE slightly improves PR-AUC, but its F1-score is lower than the standard Random Forest model. Therefore, the standard Random Forest model is selected as the best-performing model in this project.

## Key Findings

Severe class imbalance is the main challenge in this task. Since fraudulent transactions are extremely rare, accuracy is not a reliable metric for evaluating model performance.

Precision, recall, F1-score, and PR-AUC are more informative for this problem. The Logistic Regression baseline achieves high recall but suffers from many false positives.

Among the tested models, Random Forest achieves the best overall balance between fraud detection and false positive control. SMOTE slightly improves PR-AUC, but it does not clearly improve the final F1-score in this experiment.

## Contributors

### Chenyueyi Zhang

Responsible for data preprocessing, baseline model evaluation, evaluation pipeline, metric analysis, and final result interpretation.

### [Teammate Name]

Responsible for improved model training, Random Forest, MLP, SMOTE experiments, and model comparison experiments.

## Course Information

### EE 559: Machine Learning

University of Southern California  
Final Project
