# ============================================================
# Page 2 — Portfolio Dashboard
# Credit Risk Scorecard | Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load scored data ──────────────────────────────────────────
@st.cache_data
def load_data():
    ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MODELS_DIR = os.path.join(ROOT, 'models')
    df = pd.read_csv(os.path.join(MODELS_DIR, 'scored_applicants.csv'))
    return df

df = load_data()

TIER_COLORS = {
    'Low Risk'      : '#2ecc71',
    'Medium Risk'   : '#f39c12',
    'High Risk'     : '#e74c3c',
    'Very High Risk': '#8e44ad'
}
TIER_ORDER = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']

st.title("📊 Portfolio Risk Dashboard")
st.markdown("Overview of all 307,511 applicants scored by the credit scorecard model.")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────
st.subheader("📌 Key Portfolio Metrics")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Total Applicants",   f"{len(df):,}")
with k2:
    st.metric("Avg Credit Score",   f"{df['CREDIT_SCORE'].mean():.0f}")
with k3:
    st.metric("Overall Default Rate", f"{df['TARGET'].mean()*100:.2f}%")
with k4:
    low_risk_dr = df[df['RISK_TIER']=='Low Risk']['TARGET'].mean()*100
    st.metric("Low Risk Default Rate", f"{low_risk_dr:.2f}%")
with k5:
    vhr_dr = df[df['RISK_TIER']=='Very High Risk']['TARGET'].mean()*100
    st.metric("Very High Risk Default Rate", f"{vhr_dr:.2f}%")

st.markdown("---")

# ── Row 1: Score Distribution + Tier Breakdown ────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Score Distribution")
    fig1 = px.histogram(
        df, x='CREDIT_SCORE',
        color='RISK_TIER',
        color_discrete_map=TIER_COLORS,
        category_orders={'RISK_TIER': TIER_ORDER},
        nbins=60,
        title='Credit Score Distribution by Risk Tier',
        labels={'CREDIT_SCORE': 'Credit Score', 'count': 'Applicants'},
        opacity=0.85
    )
    fig1.update_layout(
        bargap=0.05,
        legend_title='Risk Tier',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='gray'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🥧 Portfolio Tier Breakdown")
    tier_counts = df['RISK_TIER'].value_counts().reindex(TIER_ORDER)
    fig2 = px.pie(
        values=tier_counts.values,
        names=tier_counts.index,
        color=tier_counts.index,
        color_discrete_map=TIER_COLORS,
        title='Applicant Distribution by Risk Tier',
        hole=0.4
    )
    fig2.update_traces(textposition='outside', textinfo='percent+label')
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='gray',
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Default Rate by Tier + Score Boxplot ───────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("🎯 Default Rate by Risk Tier")
    tier_stats = df.groupby('RISK_TIER').agg(
        Default_Rate=('TARGET', lambda x: round(x.mean()*100, 2)),
        Count=('TARGET', 'count')
    ).reindex(TIER_ORDER).reset_index()

    fig3 = px.bar(
        tier_stats,
        x='RISK_TIER', y='Default_Rate',
        color='RISK_TIER',
        color_discrete_map=TIER_COLORS,
        text='Default_Rate',
        title='Default Rate (%) per Risk Tier',
        labels={'Default_Rate': 'Default Rate (%)', 'RISK_TIER': 'Risk Tier'}
    )
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='gray',
        yaxis=dict(range=[0, 40])
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("📦 Score Range by Risk Tier")
    fig4 = px.box(
        df, x='RISK_TIER', y='CREDIT_SCORE',
        color='RISK_TIER',
        color_discrete_map=TIER_COLORS,
        category_orders={'RISK_TIER': TIER_ORDER},
        title='Credit Score Spread per Risk Tier',
        labels={'CREDIT_SCORE': 'Credit Score', 'RISK_TIER': 'Risk Tier'}
    )
    fig4.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='gray'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Score by Outcome + Tier Summary Table ──────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("⚖️ Score Distribution — Defaulted vs Repaid")
    fig5 = px.histogram(
        df, x='CREDIT_SCORE',
        color=df['TARGET'].map({0: 'Repaid', 1: 'Defaulted'}),
        color_discrete_map={'Repaid': '#3498db', 'Defaulted': '#e74c3c'},
        nbins=50, opacity=0.7, barmode='overlay',
        title='Score Overlap: Repaid vs Defaulted',
        labels={'CREDIT_SCORE': 'Credit Score', 'color': 'Outcome'}
    )
    fig5.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='gray',
        legend_title='Outcome'
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("📋 Tier Summary Table")
    summary = df.groupby('RISK_TIER').agg(
        Applicants      = ('TARGET', 'count'),
        Avg_Score       = ('CREDIT_SCORE', 'mean'),
        Min_Score       = ('CREDIT_SCORE', 'min'),
        Max_Score       = ('CREDIT_SCORE', 'max'),
        Default_Rate_Pct= ('TARGET', lambda x: round(x.mean()*100, 2))
    ).reindex(TIER_ORDER).reset_index()

    summary['Applicants']       = summary['Applicants'].apply(lambda x: f"{x:,}")
    summary['Avg_Score']        = summary['Avg_Score'].round(1)
    summary['Default_Rate_Pct'] = summary['Default_Rate_Pct'].apply(lambda x: f"{x}%")

    summary.columns = ['Risk Tier', 'Applicants', 'Avg Score',
                       'Min Score', 'Max Score', 'Default Rate']
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔢 Score Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Min Score",    df['CREDIT_SCORE'].min())
    s2.metric("Max Score",    df['CREDIT_SCORE'].max())
    s3.metric("Median Score", df['CREDIT_SCORE'].median())
    s4.metric("Std Dev",      f"{df['CREDIT_SCORE'].std():.1f}")

st.markdown("---")
st.caption("📌 Scorecard built using Points to Double Odds methodology | "
           "Home Credit Default Risk Dataset | 307,511 applicants")