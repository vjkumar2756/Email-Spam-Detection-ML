import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from datetime import datetime


st.set_page_config(
    page_title="Inbound - Enterprise Spam Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


from src.predict import SpamPredictor


@st.cache_resource
def initialize_predictor():
    try:
        return SpamPredictor(model_dir='models')
    except Exception:
        return None

predictor = initialize_predictor()


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant SaaS Gradient Header Banner */
    .gradient-header {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.2);
    }
    
    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Color Themes */
    .status-spam {
        background: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #EF4444;
        color: #F87171;
        padding: 1rem;
        border-radius: 4px;
    }
    .status-ham {
        background: rgba(34, 197, 94, 0.15);
        border-left: 5px solid #22C55E;
        color: #4ADE80;
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="gradient-header">
    <h1 style='margin:0; font-weight:700; font-size: 2.5rem;'>🛡️ Inbound AI</h1>
    <p style='margin:0.5rem 0 0 0; opacity:0.9; font-size:1.1rem;'>Enterprise NLP Gateway & Real-time Spam Diagnostics Engine</p>
</div>
""", unsafe_allow_html=True)

if predictor is None:
    st.error("🚨 Inference binaries missing. Please run `python src/train.py` from your workspace core root to compile model states before using the app.")
    st.stop()


st.sidebar.title("Configuration & Analytics")
app_mode = st.sidebar.selectbox("Choose Application Interface", ["Real-time Diagnostics", "System Performance & Metrics"])


DATASET_STATS = {
    "Total Sample Records": 5572,
    "Ham Messages Count": 4825,
    "Spam Messages Count": 747,
    "Baseline Class Split Ratio (Ham/Spam)": "86.6% / 13.4%"
}

if app_mode == "Real-time Diagnostics":
    st.subheader("✉️ Message Classification Workspace")
    
    user_input = st.text_area("Paste structural email logs or message content strings here:", height=180, 
                             placeholder="Dear customer, your bank account verification is requested immediately. Click here...")
    
    if st.button("Run Diagnostic Analysis Pipeline", type="primary"):
        if not user_input.strip():
            st.warning("Please supply valid text inputs to invoke classifier metrics.")
        else:
            with st.spinner("Executing NLP Pipeline Tokenization & Vector Scoring..."):
                # Compute predictions via core modules
                res = predictor.predict(user_input)
                
                # Contextual structural metrics derivations
                char_count = len(user_input)
                word_count = len(user_input.split())
                tokens = res["cleaned_text"].split()
                token_count = len(tokens)
                risk_score = round(res["probability_spam"] * 100, 1)
                
                # Visual Layout Segmentations split column layouts
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📊 Diagnostic Output Classification")
                    if res["label"] == "SPAM":
                        st.markdown(f"""
                        <div class='status-spam'>
                            <h3 style='margin:0;'>⚠️ THREAT DETECTED: SPAM</h3>
                            <p style='margin:0.5rem 0 0 0;'>This message exhibits strong linguistic markers matching known commercial promotional payloads or security risks.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='status-ham'>
                            <h3 style='margin:0;'>✅ SECURE: HAM (GENUINE)</h3>
                            <p style='margin:0.5rem 0 0 0;'>Linguistic telemetry signifies high authenticity. The text is safe for system distribution paths.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Statistical dynamic tracking indicators
                    st.markdown("<br>", unsafe_allow_html=True)
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Character Count", char_count)
                    m_col2.metric("Word Count", word_count)
                    m_col3.metric("Cleaned Tokens", token_count)
                    
                with col2:
                    st.markdown("### 🎯 Security Risk Distribution Score")
                    # Build interactive Gauge telemetry tracking widget
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Email Risk Score (%)", 'font': {'size': 16}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                            'bar': {'color': "#EF4444" if res["label"] == "SPAM" else "#22C55E"},
                            'bgcolor': "rgba(0,0,0,0.1)",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.2)'},
                                {'range': [40, 70], 'color': 'rgba(234, 179, 8, 0.2)'},
                                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                            ],
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Extended structural sections details expansion paths
                st.markdown("---")
                ext_col1, ext_col2 = st.columns(2)
                
                with ext_col1:
                    st.markdown("### 🔍 Advanced Text Transformation Profiling")
                    st.markdown("**Original Context Sample String:**")
                    st.caption(user_input[:300] + ("..." if len(user_input) > 300 else ""))
                    st.markdown("**Normalized Text Signature Output (Post-NLP Processing):**")
                    st.info(res["cleaned_text"] if res["cleaned_text"] else "[No active content tokens survived normalization]")
                    
                with ext_col2:
                    st.markdown("### 🔤 Keyword Extraction Token Frequency Analysis")
                    if tokens:
                        token_df = pd.DataFrame(tokens, columns=['Token']).value_counts().reset_index(name='Frequency')
                        fig_bar = px.bar(token_df.head(8), x='Frequency', y='Token', orientation='h',
                                         title="Top Cleaned Token Frequencies",
                                         color_discrete_sequence=['#06B6D4'])
                        fig_bar.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.caption("Insufficient clean structural values present to derive granular histograms charts.")
                
                # File download reporting pipelines integration framework links
                st.markdown("---")
                report_payload = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "telemetry_metrics": {"characters": char_count, "words": word_count, "tokens": token_count},
                    "risk_assessment_score": risk_score,
                    "classification_verdict": res["label"],
                    "sanitized_payload": res["cleaned_text"]
                }
                st.download_button(
                    label="📥 Download JSON System Diagnosis Report",
                    data=json.dumps(report_payload, indent=4),
                    file_name=f"spam_diagnostic_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                    mime="application/json"
                )

else:
    st.subheader("📈 Model Training Context & Dataset Analytics")
    
    stat_col1, stat_col2 = st.columns([1, 2])
    
    with stat_col1:
        st.markdown("#### Primary Dataset Distributions")
        for k, v in DATASET_STATS.items():
            st.markdown(f"**{k}:** `{v}`")
            
        st.markdown("""
        > **Production Classifier Config:**
        > * **Model Class:** Multinomial Naive Bayes
        > * **Token Extraction Vectorizer:** CountVectorizer (Bag of Words)
        > * **Test Allocation Splits:** 33% Train/Test Stratified Split
        """)
        
    with stat_col2:
        st.markdown("#### Feature Importance & Volumetric Distributions")
        pie_data = pd.DataFrame({
            'Classification Label': ['Ham (Genuine)', 'Spam (Malicious/Promo)'],
            'Count Metric Weights': [4825, 747]
        })
        fig_pie = px.pie(pie_data, values='Count Metric Weights', names='Classification Label',
                         color_discrete_sequence=['#22C55E', '#EF4444'], hole=0.4)
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)


st.markdown("""
<hr style='opacity:0.2;'>
<div style='text-align: center; opacity: 0.6; font-size: 0.85rem; padding: 1rem 0;'>
    Inbound Diagnostic Platform Framework • Configured for Scikit-Learn Production Implementations
</div>
""", unsafe_allow_html=True)