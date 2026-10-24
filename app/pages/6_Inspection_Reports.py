import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import io

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Inspection Reports | Hull AI", layout="wide")

from src.storage.parquet_reader import (
    list_available_inspections,
    read_inspection_log,
    get_inspection_summary,
    export_inspection_to_csv
)
from src.reporting.report_generator import generate_pdf_inspection_report, generate_markdown_inspection_report
from app.ui import inject_css, callout, page_header

inject_css()

page_header("📄 Inspection reports", "Query logged simulation missions (Apache Parquet) and export them as CSV or a PDF survey report.")

logs_dir = ROOT / "outputs" / "inspection_logs"
inspections = list_available_inspections(logs_dir)

if not inspections:
    callout("No inspection missions logged yet. Go to <b>Drone Simulation</b> and click 'Save Parquet Log' "
            "or run <b>python scripts/run_simulation.py</b>.")
    st.stop()

# Select inspection
insp_options = [f"{i['inspection_id']} (Records: {i['total_records']}, Coverage: {i['final_coverage_pct']:.1f}%)" for i in inspections]
selected_idx = st.selectbox("Select Logged Mission", range(len(insp_options)), format_func=lambda i: insp_options[i])
selected_insp = inspections[selected_idx]
parquet_path = Path(selected_insp["file_path"])

# Load summary and DataFrame
summary = get_inspection_summary(parquet_path)
df = read_inspection_log(parquet_path)

# KPI Cards row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mission ID", summary["inspection_id"].split("_")[-1])
c2.metric("Final Coverage", f"{summary['final_coverage_pct']:.1f}%")
c3.metric("Suspected Defects", summary["total_suspected_defects"])
c4.metric("Mean Confidence", f"{summary['mean_confidence']*100:.1f}%")

if "ground_truth_defect" in df and df["predicted_defect_class"].ne("unprocessed").any():
    gt, pr = df["ground_truth_defect"].astype(bool), df["defect_detected"].astype(bool)
    tp, fp = int((gt & pr).sum()), int((~gt & pr).sum())
    fn, tn = int((gt & ~pr).sum()), int((~gt & ~pr).sum())
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Frames correct vs ground truth", f"{(tp + tn) / len(df) * 100:.1f}%")
    g2.metric("Defect frames caught", f"{tp} / {tp + fn}")
    g3.metric("False alarms", f"{fp} / {fp + tn}")
    g4.metric("Missed defect frames", fn)
    st.caption("Ground truth is stored per frame in the log (column ground_truth_defect). Logs recorded before the "
               "simulation fix used frames without real ground truth and score poorly here.")

st.write("---")

# Download Buttons row
col_d1, col_d2, col_d3 = st.columns(3)

# 1. Parquet Download
with open(parquet_path, "rb") as f:
    parquet_bytes = f.read()
with col_d1:
    st.download_button(
        "📥 Download Apache Parquet (.parquet)",
        data=parquet_bytes,
        file_name=parquet_path.name,
        mime="application/octet-stream",
        width="stretch"
    )

# 2. CSV Download
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
with col_d2:
    st.download_button(
        "📥 Download CSV Log (.csv)",
        data=csv_buffer.getvalue(),
        file_name=f"{parquet_path.stem}.csv",
        mime="text/csv",
        width="stretch"
    )

# 3. PDF Report Generation & Download
pdf_path = ROOT / "outputs" / "reports" / f"{parquet_path.stem}_report.pdf"
generate_pdf_inspection_report(parquet_path, pdf_path)
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

with col_d3:
    st.download_button(
        "📥 Download PDF Survey Report (.pdf)",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        width="stretch"
    )

st.write("---")

# Interactive Telemetry Query Table
st.subheader("🔍 Structured PyArrow Telemetry & Defect Records")

# Filtering controls
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    filter_defects = st.checkbox("Show Defects Only", value=True)
with f_col2:
    all_zones = ["All Zones"] + sorted(list(df["zone_id"].unique()))
    selected_zone = st.selectbox("Filter by Hull Zone", all_zones)
with f_col3:
    min_conf = st.slider("Min Confidence Score", 0.0, 1.0, 0.4, 0.05)

filtered_df = df.copy()
if filter_defects:
    filtered_df = filtered_df[filtered_df["defect_detected"] == True]
if selected_zone != "All Zones":
    filtered_df = filtered_df[filtered_df["zone_id"] == selected_zone]
filtered_df = filtered_df[filtered_df["confidence_score"] >= min_conf]

st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")

display_cols = [
    "step_id", "timestamp", "drone_x", "drone_y", "drone_z", "drone_heading_deg",
    "zone_id", "camera_image_id", "predicted_defect_class", "confidence_score",
    "defect_detected", "defect_x_min", "defect_y_min", "defect_x_max", "defect_y_max",
    "inspection_status", "coverage_pct"
]
st.dataframe(filtered_df[display_cols], width="stretch")

st.write("---")
st.subheader("⚠️ Scientific & Legal Disclaimers")
callout("""
<b>1. Optical Surface NDT Scope:</b> Visual machine learning models detect surface discoloration, rust scale, and fissure cracks visible to the RGB sensor.
They cannot evaluate plate thinning, internal corrosion pitting, or subsurface metal fatigue without complementary ultrasonic thickness testing.<br/>
<b>2. Terminology Standard:</b> In accordance with DNV / Lloyd's Register NDT guidelines, all detections are cataloged as <i>'suspected defects'</i> with status <i>'requires further inspection'</i>.<br/>
<b>3. Model Confidence:</b> Confidence scores reflect model output, not calibrated structural safety factors. Certified naval architect review is mandatory before dry-dock weld scheduling.
""", "warn")
