"""Streamlit entry point.

Run with:
    streamlit run app.py
"""
import logging

import streamlit as st
import yaml

from tabs import dataset_tab, error_tab, explainability_tab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CIFAR-10 Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("CIFAR-10 Model Analysis Dashboard")
st.caption("Interactive dashboard for dataset exploration, error analysis, and model explainability.")


@st.cache_data
def load_config() -> dict:
    logger.info("Loading config.yaml")
    with open("config.yaml") as f:
        return yaml.safe_load(f)


try:
    cfg = load_config()
except FileNotFoundError:
    st.error("config.yaml not found. Make sure you run `streamlit run app.py` from `lab_06/`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["📊 Dataset Exploration", "🔬 Error Analysis", "💡 Prediction & Explainability"]
)

with tab1:
    dataset_tab.render(cfg)

with tab2:
    error_tab.render(cfg)

with tab3:
    explainability_tab.render(cfg)
