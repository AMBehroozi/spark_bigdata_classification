# Credit Default Prediction (PySpark + Parquet)

This repository contains a **big-data machine learning pipeline** for predicting credit card default risk using **Apache Spark**.
The workflow is designed for large datasets stored in **Parquet format** and uses Spark’s distributed processing to handle data at scale.

---

## 🔍 Project Summary

We work with monthly customer profiles containing anonymized financial features (delinquency, balance, payment, spend, risk indicators).
The task is to predict whether a customer will **default within 120 days** after their latest credit card statement.

---

## 🏗️ Pipeline Overview

The ML pipeline includes:

1. **Load Parquet data** using Spark
2. **Data cleaning**

   * Remove fully-null rows
   * Drop high-missing columns
3. **Vectorized imputation** for numerical and categorical features
4. **Categorical encoding** using `StringIndexer`
5. **Feature assembling** using `VectorAssembler`
6. **Scaling** using `StandardScaler`
7. **Train–test split**
8. **Class imbalance handling** with class weights
9. **Model training** using Spark ML’s `GBTClassifier`
10. **Evaluation utilities** (ROC, confusion matrix, etc.)

---

## ⚙️ Technologies Used

* Apache Spark (PySpark)
* Parquet
* Spark MLlib
* Python
* Matplotlib (for optional visualizations)

---
