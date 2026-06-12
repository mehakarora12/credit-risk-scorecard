# ============================================================
# app.py — Main Entry Point
# Credit Risk Scorecard | Streamlit App
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Landing Page ─────────────────────────────────────────────
st.title("🏦 Credit Risk Scorecard")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### What is this app?
    A bank-style credit scoring system built using the 
    **Home Credit Default Risk** dataset (Kaggle).
    
    It uses the **Points to Double Odds** methodology —
    the same technique used by **CIBIL, FICO, and Experian**
    to generate credit scores.
    
    ---
    ### How to use
    - **Page 1 — Applicant Scorer** : Enter applicant details 
      and get an instant credit score with risk tier and 
      explanation of key drivers
    - **Page 2 — Portfolio Dashboard** : Explore the full 
      dataset with KPIs, score distributions, and tier analysis
    """)

with col2:
    st.markdown("""
    ### Model Details
    | Item | Detail |
    |---|---|
    | Dataset | Home Credit Default Risk |
    | Rows | 307,511 applicants |
    | Features | 24 (WoE/IV selected) |
    | Models | Logistic Regression + XGBoost |
    | LR AUC | 0.7065 |
    | XGB AUC | 0.7391 |
    | KS Stat | 0.3524 |
    | Score Scale | 300 – 900 (CIBIL-style) |
    | Method | Points to Double Odds (PDO=20) |
    """)

st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px'>
    Built by Mahak | B.Tech CSE (AI) | 
    Data Science Portfolio Project
    </div>
""", unsafe_allow_html=True)