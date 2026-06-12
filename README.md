# 🏦 Credit Risk Scorecard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-yellow?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Demo:** [credit-risk-scorecard-b9yaqtzqdujnhofqmv5ybw.streamlit.app](https://credit-risk-scorecard-b9yaqtzqdujnhofqmv5ybw.streamlit.app)

A bank-style credit scoring system built on 307,511 applicants using the **Home Credit Default Risk** dataset (Kaggle). Implements the **Points to Double Odds** methodology — the same technique used by CIBIL, FICO, and Experian — producing a 300–900 credit score with SHAP-based explainability and an interactive Streamlit dashboard.

---

## ✨ Project Highlights

- **End-to-end pipeline** from raw data → WoE/IV feature selection → dual model training → CIBIL-style scorecard → deployed web app
- **Industry-standard methodology**: Points-to-Double-Odds (PDO=20) scorecard with monotonically calibrated risk tiers
- **Dual model architecture**: Logistic Regression for scorecard points, XGBoost for default probability + SHAP drivers
- **SMOTE applied post-split** on train set only — avoids the common data leakage mistake in imbalanced classification
- **Production-ready Streamlit app** with input validation, real-time scoring, SHAP explainability, and portfolio risk dashboard

---

## 📊 Model Performance

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC-ROC | 0.7065 | **0.7391** |
| Gini Coefficient | 0.4130 | **0.4782** |
| KS Statistic | 0.3057 | **0.3524** |

---

## 🎯 Risk Tier Breakdown

| Risk Tier | Score Range | Default Rate | Portfolio Share |
|---|---|---|---|
| 🟢 Low Risk | 650 – 900 | 2.01% | 9.1% |
| 🟡 Medium Risk | 600 – 649 | 5.26% | 59.0% |
| 🔴 High Risk | 550 – 599 | 13.72% | 29.4% |
| 🟣 Very High Risk | 300 – 549 | 30.09% | 2.5% |

---

## 🏗️ Project Architecture

```
Phase 1 — Data Understanding     → Schema analysis, missing value audit, class imbalance check
Phase 2 — EDA                    → Distribution plots, correlation analysis, target vs feature analysis
Phase 3 — Feature Engineering    → WoE/IV binning via scorecardpy, 24 features selected from 100+
Phase 4 — Modeling               → LR + XGBoost with SMOTE, hyperparameter tuning, AUC/KS/Gini eval
Phase 5 — Scorecard Creation     → Points-to-Double-Odds method, 300–900 scale, risk tier calibration
Phase 6 — Streamlit Deployment   → 3-page app with real-time scoring, SHAP drivers, portfolio dashboard
```

---

## 🔬 Methodology

### WoE / IV Feature Selection
Weight of Evidence (WoE) transforms categorical and continuous features into monotonic risk signals. Information Value (IV) ranks features by predictive power. 24 features with IV > 0.02 were selected from 100+ raw features across 3 tables.

### Points to Double Odds (PDO)
```
Score = OFFSET + FACTOR × (−log_odds)

Where:
  FACTOR = PDO / ln(2) = 20 / 0.693 = 28.85
  OFFSET = 600
  PDO    = 20  →  every 20-point drop doubles the default odds
```
This produces a 300–900 scale where **higher score = safer borrower**, consistent with CIBIL scoring.

### SMOTE for Class Imbalance
Dataset has 91.93% repaid vs 8.07% default. SMOTE applied **after train/test split** with `sampling_strategy=0.3` to avoid synthetic samples leaking into the test set.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Feature Engineering | scorecardpy (WoE/IV) |
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mehakarora12/credit-risk-scorecard.git
cd credit-risk-scorecard

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r app/requirements.txt

# 4. Run the app
cd app
streamlit run app.py
```

---

## 📁 Folder Structure

```
credit-risk-scorecard/
├── app/
│   ├── app.py                        ← Landing page
│   ├── requirements.txt
│   └── pages/
│       ├── 1_Applicant_Scorer.py     ← Credit scoring form + SHAP drivers
│       └── 2_Portfolio_Dashboard.py  ← Portfolio KPIs + Plotly charts
├── Data/
│   └── master_features.csv.gz        ← 307,511 applicants, 26 features
├── models/
│   ├── logistic_regression.pkl
│   ├── xgboost_model.json
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.json
│   ├── scorecard_params.json
│   └── scored_applicants.csv
├── Notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_scorecard.ipynb
├── .python-version
├── runtime.txt
└── README.md
```

---

*Built by Mahak | B.Tech CSE (AI) | Data Science Portfolio Project*
