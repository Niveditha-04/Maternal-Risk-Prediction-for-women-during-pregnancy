<div align="center">

# Maternal Risk Prediction during Pregnancy

**Benchmarks 9 supervised ML classifiers on IoT-collected clinical indicators to predict maternal risk level (low / mid / high) — Random Forest achieves ~89% accuracy with 10-fold cross-validation and ensemble voting.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189AB4?style=flat)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

</div>

---

## The Problem

Approximately **358,000 maternal mortalities** are recorded worldwide every year — 99% in developing countries. Pregnancy complications span a wide range of conditions with varying severity, making early risk screening critical yet difficult at scale. This project applies supervised ML to IoT-collected clinical indicators to classify each patient's pregnancy risk level so that high-risk cases can be prioritized for early intervention.

---

## Dataset — IoT-Collected Clinical Indicators

| Feature | Unit | Clinical significance |
|---------|------|-----------------------|
| `Age` | Years (at pregnancy) | Older age increases complication risk |
| `SystolicBP` | mmHg | Upper blood pressure — hypertension indicator |
| `DiastolicBP` | mmHg | Lower blood pressure — key pregnancy monitoring metric |
| `BS` | mmol/L (molar concentration) | Blood glucose level — gestational diabetes indicator |
| `HeartRate` | Beats per minute | Cardiac stress indicator |
| `RiskLevel` | **Target** | Low · Mid · High risk classification |

**Dataset:** 1,014 patient samples · 3 risk classes (low / mid / high)

---

## Approach

### Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. EDA | pandas · seaborn | Distribution plots · class balance · feature correlations |
| 2. Preprocessing | scikit-learn | Feature scaling · train/test split · class imbalance check |
| 3. Modeling | 9 classifiers | Train and evaluate each on identical 10-fold CV splits |
| 4. Ensemble | Voting Classifier | Combine top individual models for improved reliability |
| 5. Evaluation | Accuracy · confusion matrix · classification report | Compare all 9 classifiers head-to-head |

### 9 Classifiers Benchmarked

| # | Classifier | Type |
|---|-----------|------|
| 1 | Random Forest | Ensemble (bagging) |
| 2 | Gradient Boosting | Ensemble (boosting) |
| 3 | XGBoost | Optimized gradient boosting |
| 4 | CatBoost | Gradient boosting with categorical handling |
| 5 | Extra Trees | Randomized ensemble |
| 6 | Decision Tree | Single tree baseline |
| 7 | Logistic Regression | Linear baseline |
| 8 | SGD Classifier | Linear with stochastic gradient descent |
| 9 | Voting Classifier | Ensemble of top performers |

---

## Results

| Classifier | Accuracy |
|-----------|----------|
| **Random Forest** | **~89%** ← best |

| Metric | Value |
|--------|-------|
| Best accuracy | **~89%** (Random Forest) |
| Patient samples | **1,014** |
| Risk classes | **3** (low · mid · high) |
| Classifiers benchmarked | **9** |
| Validation strategy | **10-fold cross-validation** |
| Ensemble method | Voting Classifier |

---

## Demo

### Video Walkthrough
> *2-minute walkthrough: dataset exploration → 9 classifier training → accuracy comparison → ensemble results.*

[![Watch the Demo](https://img.shields.io/badge/Watch%20Demo-Coming%20Soon-red?style=for-the-badge&logo=youtube)](#)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python |
| ML | scikit-learn · XGBoost · CatBoost |
| Data wrangling | pandas · NumPy |
| Visualization | Matplotlib · Seaborn · interactive plots |
| Validation | 10-fold cross-validation |

---

## Setup & Run

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn

# Run notebook
jupyter notebook maternal_risk_prediction.ipynb
