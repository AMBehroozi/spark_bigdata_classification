# 📌 Credit Default Prediction with Apache Spark

### **Big-Data Machine Learning Pipeline Using PySpark + Parquet + Gradient Boosted Trees**

---

## 🚀 Project Overview

This repository contains an end-to-end **big-data machine learning pipeline** for predicting **credit card default events** using **Apache Spark**.
The dataset consists of **millions of rows** stored in **Parquet format**, and the goal is to predict whether a customer will **default within 120 days** after their most recent credit card statement.

This project demonstrates:

* Large-scale **feature engineering**
* Efficient **Spark DataFrame transformations**
* **Vectorized** cleaning & imputation
* **Categorical indexing**
* **Feature scaling**
* **Gradient-Boosted Trees (GBT)** for binary classification
* Evaluation with **ROC-AUC**, **confusion matrix**, **precision/recall**, and **default probability analysis**
* Big-data optimization (partitioning, caching, memory tuning)

The entire workflow is built to scale to **tens of millions of rows**, something impossible with pandas on a single machine.

---

## 📂 Dataset Description

Each row represents one **monthly customer profile** containing:

### 🔹 Feature Groups

* **D_*** — Delinquency behavior
* **S_*** — Spending patterns
* **P_*** — Payment information
* **B_*** — Balances
* **R_*** — Risk/behavioral indicators

### 🔹 Categorical Features

```
['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
```

### 🔹 Target Variable

`target = 1` means the customer **defaulted** (did not pay the due amount within 120 days).
The negative class is **downsampled to 5%**, so class weighting is required.

### 🔹 File Format

Data is provided as a large **Parquet** file (~190 columns, ~17M rows).

---

## 🏗️ Architecture of the Pipeline

### ✔️ 1. **Load Data from Parquet**

* High-speed I/O using PySpark
* Schema auto‐detection
* Smart repartitioning (optimal based on RAM & cores)

### ✔️ 2. **Data Cleaning**

* Remove rows where **all features are NULL / NaN**
* Drop columns with **>95% missing values**
* Vectorized missing-value handling (no Python loops)

### ✔️ 3. **Imputation (Vectorized Spark-Optimized)**

* Median imputation for numerical columns
* `"missing"` category for string features
* Broadcast-based implementation (much faster than Spark Imputer)

### ✔️ 4. **Categorical Encoding**

* `StringIndexer` with `handleInvalid="keep"`

### ✔️ 5. **Feature Assembly + Scaling**

* `VectorAssembler` → `features_unnorm`
* `StandardScaler` → `features`

### ✔️ 6. **Train/Test Split**

### ✔️ 7. **Handling Class Imbalance**

Inverse-frequency weighting:

[
\text{weight(class)} = \frac{N_\text{total}}{N_\text{class}}
]

### ✔️ 8. **Model Training**

Using Spark’s distributed Gradient-Boosted Trees:

* `maxDepth = 6`
* `maxIter = 60`
* `stepSize = 0.1`
* `subsamplingRate = 0.8`
* `weightCol = "class_weight"`

### ✔️ 9. **Evaluation**

We compute:

* **AUC (ROC)**
* **Confusion matrix**
* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**
* **Specificity**
* **F1 score**
* **Probability distribution plots**

### ✔️ 10. **Visualization**

* Matplotlib confusion matrix
* ROC curve
* Class-wise probability histograms

---

## 📊 Results (Example)

Confusion Matrix (on test split):

|            | Pred 0         | Pred 1        |
| ---------- | -------------- | ------------- |
| **True 0** | 686,390 (62%)  | 145,028 (13%) |
| **True 1** | 26,298  (2.4%) | 249,018 (22%) |

### Key Metrics:

* **Accuracy:** 84–85%
* **Recall (TPR):** ~90%
* **Precision:** ~63%
* **Specificity:** ~82%
* **AUC:** ~0.90+

This is a **strong baseline model**, especially given class imbalance and noisy financial data.

---

## ⚙️ Technologies Used

| Component              | Technology                           |
| ---------------------- | ------------------------------------ |
| Data storage           | Parquet                              |
| Distributed processing | Apache Spark                         |
| Language               | Python (PySpark, NumPy, Matplotlib)  |
| Modeling               | Spark MLlib (GBTClassifier)          |
| Visuals                | Matplotlib                           |
| Infrastructure         | HPC cluster / local multi-core setup |

---

## 📘 Repository Structure

```
├── data/               # Parquet dataset (not stored in repo)
├── notebooks/          # Jupyter notebooks for pipeline
├── spark_utils/        # Helper scripts (hardware detection, memory tuning)
├── models/             # Saved Spark models (optional)
├── results/            # Confusion matrix, plots, metrics
├── README.md           # This file
```

---

## 🔥 Key Strengths of This Project

* Fully vectorized **Spark-native** pipeline
* Supports **tens of millions of rows**
* High-efficiency imputation (percentile_approx + broadcast)
* Handles **skew**, **missingness**, and **class imbalance**
* Produces clean, interpretable risk predictions
* Modular and ready for **extension** (XGBoost, MLflow, hyperparameter tuning)

---

## 📈 Future Improvements

* Hyperparameter search (Spark CrossValidator)
* XGBoost4J or LightGBM distributed training
* Customer-level sequence modeling (LSTM/FNO)
* Calibration (Platt scaling / isotonic regression)
* Explainability (SHAP with sampling)

---

## 🎯 Conclusion

This repository provides a complete, scalable machine learning solution for credit risk modeling using **Apache Spark**, suitable for real-world financial-grade workloads.

If needed, I can also add:

✅ Hyperparameter tuning scripts
✅ MLflow logging
✅ Dockerfile
✅ Cluster configuration guide
✅ Fully formatted Jupyter notebook

Just tell me!
