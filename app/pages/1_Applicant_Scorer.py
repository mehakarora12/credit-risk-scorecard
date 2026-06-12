# ============================================================
# Page 1 — Applicant Scorer (Fixed — XGBoost SHAP for drivers)
# Credit Risk Scorecard | Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap

st.set_page_config(
    page_title="Applicant Scorer",
    page_icon="🎯",
    layout="wide"
)

# ── Path setup ────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(ROOT, 'models')

# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource
def load_models():
    from xgboost import XGBClassifier
    lr             = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    scaler         = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    xgb            = XGBClassifier()
    xgb.load_model(os.path.join(MODELS_DIR, 'xgboost_model.json'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, 'scorecard_params.json')) as f:
        params = json.load(f)
    return lr, scaler, label_encoders, xgb, feature_names, params

lr, scaler, label_encoders, xgb_model, feature_names, params = load_models()

FACTOR = params['FACTOR']
OFFSET = params['OFFSET']

# ── SHAP explainer (cached separately — slow to init) ─────────
@st.cache_resource
def get_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)

explainer = get_explainer(xgb_model)

# ── Helper: assign tier ───────────────────────────────────────
def assign_tier(score):
    if score >= 650:   return 'Low Risk',        '#2ecc71'
    elif score >= 600: return 'Medium Risk',     '#f39c12'
    elif score >= 550: return 'High Risk',       '#e74c3c'
    else:              return 'Very High Risk',  '#8e44ad'

# ── Helper: encode input ──────────────────────────────────────
def encode_input(input_dict):
    df_input = pd.DataFrame([input_dict])
    cat_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'CODE_GENDER']
    for col in cat_cols:
        if col in df_input.columns and col in label_encoders:
            try:
                df_input[col] = label_encoders[col].transform(
                    df_input[col].astype(str))
            except:
                df_input[col] = 0
    return df_input[feature_names]

# ── Helper: LR scorecard score ────────────────────────────────
def compute_lr_score(df_encoded):
    X_scaled  = scaler.transform(df_encoded)
    log_odds  = lr.intercept_[0] + X_scaled.dot(lr.coef_[0])
    score_raw = OFFSET + FACTOR * (-log_odds)
    return int(np.clip(score_raw, 300, 900))

# ── Helper: XGBoost SHAP drivers ─────────────────────────────
def compute_shap_drivers(df_encoded):
    sv           = explainer.shap_values(df_encoded)
    shap_series  = pd.Series(sv[0], index=feature_names)
    # positive shap = pushes toward DEFAULT = hurts score
    # we flip sign so positive = helps score (consistent with scorecard)
    shap_flipped = -shap_series
    return shap_flipped.sort_values(ascending=False)

# ── Readable feature name mapping ────────────────────────────
FEATURE_LABELS = {
    'EXT_SOURCE_1'           : 'External Credit Score 1',
    'EXT_SOURCE_2'           : 'External Credit Score 2',
    'EXT_SOURCE_3'           : 'External Credit Score 3',
    'EXT_SOURCE_MIN'         : 'Lowest External Credit Score',
    'EXT_SOURCE_STD'         : 'Credit Score Consistency',
    'AGE_YEARS'              : 'Applicant Age',
    'AGE_EMPLOYMENT_RATIO'   : 'Employment Stability Ratio',
    'AMT_INCOME_TOTAL'       : 'Annual Income',
    'AMT_CREDIT'             : 'Loan Amount',
    'AMT_ANNUITY'            : 'Monthly EMI',
    'CREDIT_INCOME_RATIO'    : 'Loan-to-Income Ratio',
    'ANNUITY_INCOME_RATIO'   : 'EMI-to-Income Ratio',
    'CREDIT_TERM'            : 'Loan Term (EMI/Loan)',
    'REGION_RATING_CLIENT'   : 'Region Risk Rating',
    'DOCUMENTS_SUBMITTED'    : 'Documents Submitted',
    'BUREAU_TOTAL_DEBT'      : 'Total Outstanding Debt',
    'BUREAU_AVG_OVERDUE'     : 'Avg Overdue Amount',
    'BUREAU_MAX_OVERDUE_DAYS': 'Max Days Overdue',
    'BUREAU_ACTIVE_RATIO'    : 'Active Loan Ratio',
    'PREV_REFUSED_COUNT'     : 'Previous Loan Rejections',
    'PREV_APPROVAL_RATE'     : 'Previous Approval Rate',
    'SOCIAL_CIRCLE_DEFAULT'  : 'Social Circle Defaults',
    'ID_PUBLISH_YEARS'       : 'Years Since ID Issued',
    'REGISTRATION_YEARS'     : 'Years at Current Address',
    'NAME_INCOME_TYPE'       : 'Employment Type',
    'NAME_EDUCATION_TYPE'    : 'Education Level',
    'NAME_FAMILY_STATUS'     : 'Family Status',
    'CODE_GENDER'            : 'Gender',
}

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════
st.title("🎯 Applicant Credit Scorer")
st.markdown("Fill in the applicant details below to generate a credit score.")
st.markdown("---")

with st.form("scorer_form"):
    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age (years)", 18, 70, 35,
            help="Your current age in years")
        code_gender = st.selectbox("Gender", ["M", "F"],
            help="M = Male, F = Female")
        name_family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married",
             "Civil marriage", "Separated", "Widow"],
            help="Your current marital/family status")

    with col2:
        name_income_type = st.selectbox(
            "Employment Type",
            ["Working", "Commercial associate",
             "Pensioner", "State servant", "Unemployed"],
            help="Working = salaried | Commercial = self-employed | "
                 "Pensioner = retired | State servant = government employee")
        name_education_type = st.selectbox(
            "Highest Education",
            ["Secondary / secondary special", "Higher education",
             "Incomplete higher", "Lower secondary", "Academic degree"],
            help="Your highest completed level of education")
        amt_income_total = st.number_input(
            "Annual Income (₹)", min_value=10000,
            max_value=10000000, value=500000, step=10000,
            help="Total yearly income from all sources before tax")

    with col3:
        region_rating = st.selectbox(
            "Region Rating", [1, 2, 3], index=1,
            help="1 = Low risk area (metro), 2 = Medium, 3 = High risk area")
        documents_submitted = st.slider(
            "Documents Submitted", 0, 10, 5,
            help="Number of supporting documents provided")
        employment_years = st.slider(
            "Years Employed", 0, 40, 5,
            help="Years at current/most recent job. Enter 0 if unemployed")

    st.markdown("---")
    st.subheader("💰 Loan Information")
    col4, col5, col6 = st.columns(3)

    with col4:
        amt_credit = st.number_input(
            "Loan Amount (₹)", min_value=10000,
            max_value=5000000, value=500000, step=10000,
            help="Total loan amount you are applying for")
        amt_annuity = st.number_input(
            "Monthly EMI (₹)", min_value=1000,
            max_value=200000, value=15000, step=1000,
            help="Expected monthly repayment. Should be < 40% of monthly income")

    with col5:
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5, 0.01,
            help="Credit score from bureau (0 = very poor, 1 = excellent)")
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5, 0.01,
            help="Second credit score source. Higher = better")
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5, 0.01,
            help="Third credit score. Higher = better. Use 0.5 if unknown")

    with col6:
        bureau_total_debt    = st.number_input("Total Outstanding Debt (₹)",
            0, 5000000, 0, 10000,
            help="Total amount owed across all existing loans")
        bureau_avg_overdue   = st.number_input("Avg Overdue Amount (₹)",
            0, 500000, 0, 1000,
            help="Average overdue amount. Enter 0 if never missed a payment")
        bureau_max_overdue   = st.number_input("Max Days Overdue",
            0, 1000, 0, 10,
            help="Max days ever late on a payment. Enter 0 if always on time")
        bureau_active_ratio  = st.slider("Active Loan Ratio", 0.0, 1.0, 0.2, 0.01,
            help="Active loans ÷ Total loans ever. Example: 2 of 5 = 0.40")

    st.markdown("---")
    st.subheader("📋 Credit History")
    col7, col8 = st.columns(2)

    with col7:
        prev_approved = st.slider("Previously Approved Applications", 0, 20, 3,
            help="How many loan applications were approved in the past")
        prev_refused  = st.slider("Previously Rejected Applications", 0, 20, 0,
            help="How many loan applications were rejected. More = higher risk")

    with col8:
        id_publish_years   = st.slider("Years Since ID Issued", 0, 20, 5,
            help="How many years ago was your government ID issued")
        registration_years = st.slider("Years at Current Address", 0, 30, 5,
            help="How long you have lived at your current address")
        social_circle_def  = st.slider("Social Circle Defaults", 0, 20, 0,
            help="People in your network who defaulted on loans. Enter 0 if unknown")

    submitted = st.form_submit_button(
        "🔍 Generate Credit Score", use_container_width=True)

# ── AFTER FORM SUBMISSION ─────────────────────────────────────
if submitted:

    # ── Validation ────────────────────────────────────────────
    errors = []
    monthly_income = amt_income_total / 12

    if amt_annuity > monthly_income * 0.6:
        errors.append(
            f"⚠️ Monthly EMI (₹{amt_annuity:,}) exceeds 60% of monthly income "
            f"(₹{monthly_income:,.0f}). Please reduce EMI or increase income.")
    if amt_credit > amt_income_total * 15:
        errors.append(
            f"⚠️ Loan amount (₹{amt_credit:,}) is more than 15x annual income. "
            f"This is unrealistic.")
    if name_income_type == 'Unemployed' and amt_credit > 200000:
        errors.append(
            "⚠️ Unemployed applicants cannot apply for loans above ₹2,00,000.")
    if employment_years > (age - 16):
        errors.append(
            f"⚠️ Years employed ({employment_years}) cannot exceed "
            f"working-age years ({age - 16}).")

    if errors:
        for e in errors:
            st.warning(e)
        st.error("❌ Please fix the above issues before generating a score.")
        st.stop()

    # ── Derived Features ──────────────────────────────────────
    employment_years_val = employment_years
    age_emp_ratio        = (employment_years_val / age
                            if (employment_years_val > 0 and age > 0)
                            else 0.0)
    credit_income_ratio  = amt_credit / amt_income_total
    annuity_income_ratio = amt_annuity / amt_income_total
    credit_term          = amt_annuity / amt_credit if amt_credit > 0 else 0
    ext_source_min       = min(ext_source_1, ext_source_2, ext_source_3)
    ext_source_std       = float(np.std([ext_source_1, ext_source_2, ext_source_3]))
    prev_total           = prev_approved + prev_refused
    prev_approval_rate   = prev_approved / prev_total if prev_total > 0 else 0.0

    input_dict = {
        'AGE_YEARS'              : age,
        'AMT_INCOME_TOTAL'       : amt_income_total,
        'AMT_CREDIT'             : amt_credit,
        'AMT_ANNUITY'            : amt_annuity,
        'REGION_RATING_CLIENT'   : region_rating,
        'DOCUMENTS_SUBMITTED'    : documents_submitted,
        'EXT_SOURCE_1'           : ext_source_1,
        'EXT_SOURCE_2'           : ext_source_2,
        'EXT_SOURCE_3'           : ext_source_3,
        'EXT_SOURCE_MIN'         : ext_source_min,
        'EXT_SOURCE_STD'         : ext_source_std,
        'AGE_EMPLOYMENT_RATIO'   : age_emp_ratio,
        'CREDIT_INCOME_RATIO'    : credit_income_ratio,
        'ANNUITY_INCOME_RATIO'   : annuity_income_ratio,
        'CREDIT_TERM'            : credit_term,
        'BUREAU_TOTAL_DEBT'      : bureau_total_debt,
        'BUREAU_AVG_OVERDUE'     : bureau_avg_overdue,
        'BUREAU_MAX_OVERDUE_DAYS': bureau_max_overdue,
        'BUREAU_ACTIVE_RATIO'    : bureau_active_ratio,
        'PREV_REFUSED_COUNT'     : prev_refused,
        'PREV_APPROVAL_RATE'     : prev_approval_rate,
        'SOCIAL_CIRCLE_DEFAULT'  : social_circle_def,
        'ID_PUBLISH_YEARS'       : id_publish_years,
        'REGISTRATION_YEARS'     : registration_years,
        'NAME_INCOME_TYPE'       : name_income_type,
        'NAME_EDUCATION_TYPE'    : name_education_type,
        'NAME_FAMILY_STATUS'     : name_family_status,
        'CODE_GENDER'            : code_gender,
    }

    # ── Encode + Score ────────────────────────────────────────
    df_encoded = encode_input(input_dict)
    score      = compute_lr_score(df_encoded)
    tier, color= assign_tier(score)

    # ── XGBoost default probability ───────────────────────────
    xgb_prob   = xgb_model.predict_proba(df_encoded)[0][1]

    # ── SHAP drivers from XGBoost ─────────────────────────────
    shap_drivers = compute_shap_drivers(df_encoded)

    # ── RESULT DISPLAY ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Credit Score Result")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown(f"""
        <div style='text-align:center; padding:30px;
                    background:{color}22; border-radius:15px;
                    border: 2px solid {color}'>
            <h1 style='color:{color}; font-size:72px; margin:0'>{score}</h1>
            <p style='font-size:16px; color:gray; margin:5px'>out of 900</p>
            <h3 style='color:{color}; margin:10px'>{tier}</h3>
            <p style='color:gray; font-size:13px; margin:0'>
                Default Probability: {xgb_prob*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Score Position (300–900)**")
        pct = (score - 300) / 600
        st.progress(pct)
        st.markdown(f"""
| Band | Range | Status |
|---|---|---|
| 🟢 Low Risk | 650–900 | {'✅ **YOU ARE HERE**' if tier=='Low Risk' else ''} |
| 🟡 Medium Risk | 600–649 | {'✅ **YOU ARE HERE**' if tier=='Medium Risk' else ''} |
| 🔴 High Risk | 550–599 | {'✅ **YOU ARE HERE**' if tier=='High Risk' else ''} |
| 🟣 Very High Risk | 300–549 | {'✅ **YOU ARE HERE**' if tier=='Very High Risk' else ''} |
        """)

    with c3:
        recommendations = {
            'Low Risk'      : ('✅ Approve',
                               'Low default probability. Approve with standard terms.'),
            'Medium Risk'   : ('📋 Manual Review',
                               'Moderate risk. Review income stability and debt levels.'),
            'High Risk'     : ('⚠️ Caution',
                               'High default risk. Consider collateral or co-applicant.'),
            'Very High Risk': ('❌ Reject',
                               'Very high default probability. Recommend rejection.')
        }
        rec_label, rec_text = recommendations[tier]
        st.markdown(f"""
        <div style='padding:20px; background:transparent;
                    border-radius:10px; border:2px solid {color};
                    border-left:8px solid {color}'>
            <p style='color:gray; margin:0; font-size:13px'>Bank Recommendation</p>
            <h3 style='color:{color}; margin:8px 0 4px 0'>{rec_label}</h3>
            <p style='margin:0'>{rec_text}</p>
            <p style='color:gray; font-size:12px; margin-top:8px'>
                XGBoost Default Probability: <b>{xgb_prob*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Score Drivers ────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Score Drivers — Why this score?")
    st.caption("Based on XGBoost SHAP values — shows true feature impact direction")

    # Features to always hide (not meaningful to display)
    HIDE_FEATURES = set()
    if employment_years_val == 0 or name_income_type == 'Unemployed':
        HIDE_FEATURES.add('AGE_EMPLOYMENT_RATIO')

    # Filter and get top drivers
    filtered = shap_drivers[~shap_drivers.index.isin(HIDE_FEATURES)]
    top_helping = filtered[filtered > 0].head(4)
    top_hurting = filtered[filtered < 0].tail(4).sort_values()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**✅ Factors helping your score:**")
        if len(top_helping) > 0:
            for feat, val in top_helping.items():
                label = FEATURE_LABELS.get(feat, feat)
                st.markdown(f"- **{label}** → +{val:.3f} impact")
        else:
            st.markdown("- No strong positive factors found")

    with col_b:
        st.markdown("**⚠️ Factors hurting your score:**")
        if len(top_hurting) > 0:
            for feat, val in top_hurting.items():
                label = FEATURE_LABELS.get(feat, feat)
                st.markdown(f"- **{label}** → {val:.3f} impact")
        else:
            st.markdown("- No strong negative factors found")

    # ── What to improve ───────────────────────────────────────
    if tier in ['High Risk', 'Very High Risk']:
        st.markdown("---")
        st.subheader("💡 How to Improve Your Score")
        tips = []
        if ext_source_min < 0.4:
            tips.append("📈 Improve your credit history — pay all bills and EMIs on time")
        if prev_refused > 2:
            tips.append("🚫 Avoid applying for multiple loans in a short period")
        if bureau_avg_overdue > 0:
            tips.append("💳 Clear all overdue payments immediately")
        if annuity_income_ratio > 0.3:
            tips.append("📉 Reduce your loan EMI — it's too high relative to your income")
        if bureau_total_debt > amt_income_total:
            tips.append("💰 Pay down existing debt before applying for a new loan")
        if not tips:
            tips.append("📊 Build a stronger credit history over the next 6-12 months")
        for tip in tips:
            st.info(tip)

    st.markdown("---")
    st.caption("⚠️ Score uses Points-to-Double-Odds method (LR). "
               "Drivers use XGBoost SHAP for accurate feature attribution.")