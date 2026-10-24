"""Autonomous Underwater ROV Ship Hull Inspection Dashboard - entry point and navigation."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Hull Inspection AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui import inject_css, sidebar_status

inject_css()

pages = {
    "Start": [
        st.Page("pages/0_Home.py", title="Home", icon=":material/home:", default=True),
        st.Page("pages/1_Overview.py", title="Project overview", icon=":material/dashboard:"),
    ],
    "Inspect": [
        st.Page("pages/2_Image_Inspection.py", title="Image inspection", icon=":material/image_search:"),
        st.Page("pages/3_Drone_Simulation.py", title="Drone simulation", icon=":material/sailing:"),
    ],
    "Evaluate": [
        st.Page("pages/5_Model_Evaluation.py", title="Model evaluation", icon=":material/monitoring:"),
        st.Page("pages/4_Dataset_Explorer.py", title="Dataset explorer", icon=":material/photo_library:"),
    ],
    "Operate": [
        st.Page("pages/6_Inspection_Reports.py", title="Inspection reports", icon=":material/description:"),
    ],
}

nav = st.navigation(pages)
st.sidebar.markdown("### 🌊 Hull Inspection AI")
st.sidebar.caption("Underwater ROV corrosion detection · FOML project")
sidebar_status()
nav.run()
