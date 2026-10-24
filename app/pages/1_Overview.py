"""Streamlit Page 1: Overview of data, splits, models and logged missions."""

import streamlit as st
from pathlib import Path
import json
import sys
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Overview | Hull Inspection AI", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import inject_css, callout, load_metrics, load_training_summary, missing_metrics_message, page_header

inject_css()
page_header("📊 Project overview", "Datasets, the leakage-free split, the two models and their held-out results.")

metrics = load_metrics("mavecodd")
summary = load_training_summary("mavecodd")

# Dataset statistics straight from the split manifests
def manifest_rows(name):
    f = ROOT / "data" / "processed" / f"split_manifest_{name}.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    return [{**s, "split": split} for split, v in data.items() for s in v["samples"]]

mav = manifest_rows("mavecodd")
crack = manifest_rows("crack")

logs_dir = ROOT / "outputs" / "inspection_logs"
parquet_files = list(logs_dir.glob("*.parquet")) if logs_dir.exists() else []
max_cov = 0.0
for p in parquet_files:
    try:
        max_cov = max(max_cov, float(pd.read_parquet(p, columns=["coverage_pct"])["coverage_pct"].max()))
    except Exception:
        pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("MaVeCoDD image/mask pairs", len(mav) if mav else "not downloaded")
c2.metric("Underwater Crack pairs", len(crack) if crack else "not downloaded")
c3.metric("Logged simulation missions", len(parquet_files))
c4.metric("Best hull coverage in a mission", f"{max_cov:.1f}%")

st.write("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🗂️ Train / validation / test split (MaVeCoDD)")
    if mav:
        df = pd.DataFrame(mav)
        tab = (df.groupby(["split", "resolution_tier"]).size().unstack(fill_value=0)
               .reindex(["train", "val", "test"]))
        tab["Scene groups"] = df.groupby("split")["scene_id"].nunique().reindex(tab.index)
        tab["Images"] = tab.drop(columns="Scene groups").sum(axis=1)
        st.dataframe(tab, width="stretch")
        st.caption("A scene group is a run of consecutive frame numbers (near-duplicates). Each group is assigned to "
                   "exactly one split, and the split is stratified by resolution tier.")
    else:
        callout("Dataset not prepared. Put the dataset in <b>data/raw/mavecodd</b> and run "
                "<b>python scripts/prepare_dataset.py</b>.", "warn")

    st.subheader("⚙️ Models")
    if metrics and summary:
        b, u = metrics["baseline_model_a"]["test"], metrics["deep_learning_model_b"]["test"]
        cv = summary["baseline"]["cv"]["chosen"]
        st.markdown(f"""
- **Model A – traditional ML**: 47 handcrafted features (Lab colour relative to the whole image, LBP texture, gradient statistics) → RBF-SVM (C = {cv['C']}{', PCA ' + str(cv['pca_components']) if cv['pca_components'] else ''}),
  chosen by scene-grouped cross-validation. Classifies 256 px tiles.
  Test tile balanced accuracy **{b['balanced_accuracy'] * 100:.1f}%**, defect recall **{b['recall'] * 100:.1f}%**.
- **Model B – deep learning**: compact U-Net ({summary['unet']['n_params'] / 1e6:.1f}M parameters), BCE + Dice loss,
  early stopping (kept epoch {summary['unet']['best_epoch']}). Predicts a corrosion mask for every pixel.
  Test Dice **{u['mean_dice']:.3f}**, test tile balanced accuracy **{u['patch_classification']['balanced_accuracy'] * 100:.1f}%**,
  defect recall **{u['patch_classification']['recall'] * 100:.1f}%**.
        """)
    else:
        missing_metrics_message()

with col_b:
    st.subheader("📈 Held-out test performance (same tiles)")
    if metrics:
        b, u = metrics["baseline_model_a"]["test"], metrics["deep_learning_model_b"]["test"]["patch_classification"]
        comp = pd.DataFrame([
            {"Metric": k, "Model A (features+SVM)": b[m], "Model B (U-Net)": u[m]}
            for k, m in [("Balanced acc.", "balanced_accuracy"), ("Precision", "precision"),
                         ("Recall", "recall"), ("F1", "f1_score")]
        ])
        fig = px.bar(comp, x="Metric", y=["Model A (features+SVM)", "Model B (U-Net)"], barmode="group",
                     color_discrete_sequence=["#3498db", "#27ae60"])
        fig.update_layout(yaxis_range=[0, 1.05], legend_title="Model", height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        of = metrics["overfitting_check"]
        st.markdown(
            f"**Overfitting check** (train → test): Model A {of['baseline_patch_balanced_accuracy']['train']:.2f} → "
            f"{of['baseline_patch_balanced_accuracy']['test']:.2f} · Model B Dice {of['unet_dice']['train']:.2f} → "
            f"{of['unet_dice']['test']:.2f}. Details on the **Model Evaluation** page.")

st.write("---")
callout("<b>Marine surveying context:</b> in hull inspection a <b>false negative</b> (missed defect) is far more "
        "costly than a false alarm. Model results are screening aids; every flagged area is a <i>suspected</i> "
        "defect that a certified surveyor must confirm.", "neutral")
