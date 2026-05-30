# Statlog German Credit Risk Modeling: Imbalanced Classification Benchmarking

This repository contains an end-to-end machine learning pipeline for evaluating credit risk using the **Statlog German Credit Dataset**. The project implements a rigorous preprocessing and benchmarking pipeline comparing four state-of-the-art tree-based ensemble algorithms to optimize default prediction under severe class imbalance.

---

## 📌 1. Context: The Business Problem

* **The Situation:** Financial institutions rely heavily on credit scoring frameworks to evaluate applicant creditworthiness. In this case, we utilize the classic Statlog German Credit Dataset (1,000 applicants) to classify borrowers into risky or non-risky categories.
* **The Problem:** Misclassifying a high-risk applicant (*Bad Credit*) costs significantly more than turning away a creditworthy customer. Traditional classifiers optimize for global accuracy, failing to capture complex non-linear default risk.
* **The Challenge:** The dataset suffers from a significant class imbalance: **70% Good Credit (700 rows) vs 30% Bad Credit (300 rows)**. Standard models blindly favor the majority class, leading to high default leakages that threaten bank liquidity.
* **The Initiative:** Deploying and benchmarking advanced tree-based boosting architectures, specifically **AdaBoost, CatBoost, XGBoost, and LightGBM**. to leverage cost-sensitive learning and error-driven optimization to safely boost minority class recall.

---

## 🛠️ 2. Action: Engineering & Modular Pipeline

To prevent data leakage and guarantee technical reproducibility, the notebook executes the following precise machine learning workflow:

* **Target Rectification:** Handled the custom target values by mapping `{1: 0, 2: 1}` to clearly isolate `0` as Good Credit and `1` as Bad Credit. 
* **Data Split Integrity:** Stratified the dataset into an **80/20 train-test split** using `train_test_split(..., test_size=0.2, random_state=42)` to maintain baseline class distribution.
* **Feature Encoding Isolation:** Detected categorical strings (`select_dtypes(include=['str'])`) and applied Scikit-Learn's `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`. The encoder was strictly `.fit_transform()` on the training set and `.transform()` on the test set to completely avoid category data leakage.
* **Cost-Sensitive Boosting:** Instead of synthetic data generation (SMOTE), internal algorithmic regularization was utilized. Specifically, LightGBM was configured with `is_unbalance=True` to penalize minority class misclassifications automatically.

---

## 📊 3. Result: Technical Evaluation & Benchmark

All four models were evaluated on the holdout test set ($N=200$). Below is the comprehensive technical breakdown of their predictive capabilities:

| Model Architecture | Global Accuracy | Class 1 (Bad) Precision | Class 1 (Bad) Recall | Test ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| **AdaBoost** | 77% | **0.67** | 41% | 0.8137 |
| **CatBoost** | **79%** | 0.76 | 42% | **0.8250** |
| **XGBoost** | 74% | 0.58 | 51% | 0.7919 |
| **LightGBM**| 73% | 0.54 | **63%** | 0.7942 |

### 📈 ROC Curve Trajectory Analysis
The models demonstrate strong discriminatory capabilities overall, with CatBoost leading the global separation metric at an **AUC of 0.8250**, followed closely by AdaBoost at **0.8137**.
