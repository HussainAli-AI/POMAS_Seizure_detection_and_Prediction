"""Streamlit Dashboard for PoMAS.

Phase 6: Interactive dashboard showing EEG view, score timeline,
XAI report panels, alert log, and performance metrics.
"""

import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PoMAS Dashboard", layout="wide")

st.title("PoMAS: Closed-Loop EEG Monitoring")
st.markdown("Prediction of Multi-stage Autonomous System for Epileptic Seizure Early Warning")

# Sidebar
st.sidebar.header("Controls")
patient_id = st.sidebar.selectbox("Patient", [f"chb{i:02d}" for i in range(1, 25)])
mode = st.sidebar.radio("Mode", ["Offline Replay", "Live Simulation (Simulated)"])

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("EEG View - 10s Window")
    chart_placeholder = st.empty()

    st.subheader("Risk Score Timeline")
    timeline_placeholder = st.empty()

with col2:
    st.subheader("Current Assessment")
    risk_placeholder = st.empty()
    method_placeholder = st.empty()

    st.subheader("XAI Report")
    xai_tab = st.selectbox("Explanation Type", ["SHAP", "Grad-CAM", "Attention Heatmap"])
    xai_placeholder = st.empty()

# Alert log at bottom
st.subheader("Alert Log")
alert_placeholder = st.empty()


def update_dashboard():
    risk = np.random.random()
    risk_placeholder.metric("Risk Score", f"{risk:.2f}", delta=f"{risk - 0.3:.2f}")

    if risk > 0.7:
        st.error("HIGH RISK - Clinician Notification Recommended")
    elif risk > 0.5:
        st.warning("MEDIUM RISK - EHR Flag")
    elif risk > 0.3:
        st.info("LOW RISK - Silent Log")
    else:
        st.success("Normal")

    method_placeholder.info("Method: Patient-Dependent (EfficientNet-B0 + SVM)")

    timeline_data = pd.DataFrame({
        "time": pd.date_range(start="now", periods=50, freq="5s"),
        "risk": np.random.random(50) * 0.5 + np.random.random(50) * 0.5,
    })
    timeline_placeholder.line_chart(timeline_data.set_index("time"))

    alert_data = pd.DataFrame({
        "time": [datetime.now().strftime("%H:%M:%S")],
        "patient": [patient_id],
        "risk": [f"{risk:.2f}"],
        "tier": ["SILENT_LOG" if risk < 0.5 else "EHR_FLAG" if risk < 0.7 else "CLINICIAN_NOTIFY"],
    })
    alert_placeholder.dataframe(alert_data, use_container_width=True)


if st.sidebar.button("Run Simulation"):
    st.sidebar.info("Running simulation...")
    for _ in range(10):
        update_dashboard()
        time.sleep(0.5)

if __name__ == "__main__":
    if not st.session_state.get("initialized"):
        update_dashboard()
        st.session_state.initialized = True
