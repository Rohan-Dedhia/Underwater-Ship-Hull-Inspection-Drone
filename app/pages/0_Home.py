"""Home page: what the project does, headline results, and a guided tour of the dashboard."""

import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Home | Hull Inspection AI", page_icon="🌊", layout="wide")

from app.ui import inject_css, callout, kpi_card, page_header, step_number, load_metrics, missing_metrics_message

inject_css()
page_header(
    "🌊 AI-Powered Underwater Ship Hull Inspection",
    "An inspection drone (ROV) photographs a ship's hull; two machine-learning models find corrosion in the "
    "photos, and every result is checked against human-labelled ground truth.",
    chips=["MaVeCoDD dataset · 76 photos", "U-Net + SVM", "Scene-grouped train/val/test", "Parquet telemetry"],
)

metrics = load_metrics("mavecodd")
logs_dir = ROOT / "outputs" / "inspection_logs"
num_missions = len(list(logs_dir.glob("*.parquet"))) if logs_dir.exists() else 0

c1, c2, c3, c4 = st.columns(4)
if metrics:
    b_test = metrics["baseline_model_a"]["test"]
    u_test = metrics["deep_learning_model_b"]["test"]
    up = u_test["patch_classification"]
    with c1:
        kpi_card(f"{u_test['mean_dice']:.2f}", "U-Net test Dice (pixel overlap)")
    with c2:
        kpi_card(f"{up['false_negatives']} vs {b_test['false_negatives']}",
                 f"Missed defect tiles of {up['n_defect_patches']}: U-Net vs SVM")
    with c3:
        kpi_card(f"{up['balanced_accuracy'] * 100:.0f}% vs {b_test['balanced_accuracy'] * 100:.0f}%",
                 "Test tile balanced accuracy")
else:
    for col in (c1, c2, c3):
        with col:
            kpi_card("—", "Run evaluation")
with c4:
    kpi_card(str(num_missions), f"Logged missions · {'GPU' if torch.cuda.is_available() else 'CPU'}")

st.write("")
if not metrics:
    missing_metrics_message()

st.subheader("Start here: a 3-step tour")
t1, t2, t3 = st.columns(3)
with t1:
    with st.container(border=True, height="stretch"):
        st.markdown(f"{step_number(1)} **Is anything overfitted?**", unsafe_allow_html=True)
        st.markdown("Each model is scored on its training images *and* on images it never saw. "
                    "Close numbers = it learned real corrosion features, not the photos.")
        st.page_link("pages/5_Model_Evaluation.py", label="Open the overfitting check", icon=":material/monitoring:")
with t2:
    with st.container(border=True, height="stretch"):
        st.markdown(f"{step_number(2)} **Check a prediction yourself**", unsafe_allow_html=True)
        st.markdown("Pick a photo tagged **TEST**. You get the human label, the model's mask and an error map "
                    "(green = found, red = false alarm, blue = missed).")
        st.page_link("pages/2_Image_Inspection.py", label="Inspect an image", icon=":material/image_search:")
with t3:
    with st.container(border=True, height="stretch"):
        st.markdown(f"{step_number(3)} **Fly the drone**", unsafe_allow_html=True)
        st.markdown("Run the ROV over a hull with 5 corrosion areas. Every camera frame is scored against its "
                    "ground truth on a live scorecard.")
        st.page_link("pages/3_Drone_Simulation.py", label="Start the simulation", icon=":material/sailing:")

st.write("")
left, right = st.columns([3, 2])
with left:
    st.subheader("How it works")
    steps = [
        ("📷", "Data", "76 dry-dock hull photos with pixel-level corrosion masks, split 60/20/20 by scene so "
                       "near-duplicate video frames never cross from training into testing."),
        ("🧮", "Model A · SVM", "47 colour/texture features per 256 px tile → RBF-SVM decides defect vs sound."),
        ("🧠", "Model B · U-Net", "A 2M-parameter network that colours every corroded pixel; trained with heavy "
                                  "augmentation and early stopping."),
        ("🤖", "Simulation", "ROV lawnmower path over a 50 m × 10 m hull with underwater optics and sensors."),
        ("🗄️", "Logs & reports", "Every frame is logged to Parquet; missions export to CSV and PDF."),
    ]
    for icon, name, text in steps:
        st.markdown(f"**{icon} {name}** — {text}")
with right:
    st.subheader("Results at a glance")
    if metrics:
        of = metrics["overfitting_check"]
        st.markdown(f"""
| | Train | Val | Test |
|---|---|---|---|
| U-Net Dice | {of['unet_dice']['train']:.2f} | {of['unet_dice']['val']:.2f} | {of['unet_dice']['test']:.2f} |
| U-Net tile bal. acc | {of['unet_patch_balanced_accuracy']['train']:.2f} | {of['unet_patch_balanced_accuracy']['val']:.2f} | {of['unet_patch_balanced_accuracy']['test']:.2f} |
| SVM tile bal. acc | {of['baseline_patch_balanced_accuracy']['train']:.2f} | {of['baseline_patch_balanced_accuracy']['val']:.2f} | {of['baseline_patch_balanced_accuracy']['test']:.2f} |
""")
        st.caption("The test split has only 21 clearly sound tiles, so test tile scores carry wide confidence "
                   "intervals (see Model evaluation).")
    callout("<b>Engineering notice:</b> detections are <i>suspected defects</i> for a certified surveyor to "
            "confirm. Cameras cannot measure plate thinning or subsurface fatigue — that needs ultrasonic testing.",
            "warn")
