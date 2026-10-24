"""Streamlit Page 5: Model evaluation, overfitting check, and Model A vs Model B comparison."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Model Evaluation | Hull AI", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import (inject_css, callout, badge, verdict_kind, load_metrics, load_training_summary,
                    missing_metrics_message, page_header)

inject_css()
page_header("📈 Model evaluation",
            "Model A (handcrafted features + SVM) versus Model B (U-Net) on scene-grouped train / validation / "
            "test splits. Every number here comes from scripts/evaluate_model.py.")

metrics = load_metrics("mavecodd")
summary = load_training_summary("mavecodd")
if not metrics:
    missing_metrics_message()
    st.stop()

base = metrics["baseline_model_a"]
unet = metrics["deep_learning_model_b"]
of = metrics["overfitting_check"]
proto = metrics["evaluation_protocol"]
sizes = proto["split_sizes"]
bt, ut = base["test"], unet["test"]["patch_classification"]

# ---------------- Headline ----------------
h1, h2, h3, h4 = st.columns(4)
h1.metric("U-Net test Dice", f"{unet['test']['mean_dice']:.3f}",
          help="Pixel overlap between predicted and true corrosion on unseen test images")
h2.metric("Missed defect tiles", f"{ut['false_negatives']} vs {bt['false_negatives']}",
          help=f"Out of {bt['n_defect_patches']} defect tiles in the test set: U-Net vs SVM")
h3.metric("Tile balanced accuracy", f"{ut['balanced_accuracy']:.2f} vs {bt['balanced_accuracy']:.2f}",
          help="U-Net vs SVM on the same test tiles")
h4.metric("U-Net overfitting", of["unet_dice"]["verdict"].replace(" meaningful", ""),
          help="Train vs held-out Dice gap")

with st.expander("ℹ️ How this evaluation works (read first)"):
    callout(f"""
<ul>
<li><b>Splits:</b> {sizes['train']} training, {sizes['val']} validation and {sizes['test']} test images. Images are
grouped into <i>scenes</i> of consecutive frames, and a whole scene goes to one split, so a near-identical neighbouring
frame can never be in training while its twin is in the test set.</li>
<li><b>Same task for both models:</b> every image is cut into 256 px tiles. A tile is <i>defect</i> if ≥ 5% of it is
corrosion and <i>sound</i> if ≤ 0.5%. Both models are scored on exactly these tiles.</li>
<li><b>Why not "image has a defect: yes/no"?</b> Every MaVeCoDD photo contains corrosion, so that score would be 100%
for a model that always says "defect" — it measures nothing.</li>
<li><b>No test-set tuning:</b> hyperparameters, early stopping and decision thresholds are chosen using training/validation
data only. The test split is used once, for the final numbers.</li>
</ul>
""", "info")

tab_of, tab_cmp, tab_seg, tab_rob, tab_safe = st.tabs(
    ["🧪 Overfitting check", "⚖️ Model A vs B", "🎨 Segmentation", "🌊 Robustness", "🛟 Safety & limits"])

# ---------------- 1. Overfitting check ----------------
with tab_of:
    rows = []
    for key, name in [("baseline_patch_balanced_accuracy", "Model A · tile balanced accuracy"),
                      ("unet_patch_balanced_accuracy", "Model B · tile balanced accuracy"),
                      ("unet_dice", "Model B · segmentation Dice")]:
        v = of[key]
        ci = v.get("test_ci95") or [None, None]
        rows.append({"Metric": name, "Train": v["train"], "Validation": v["val"], "Test": v["test"],
                     "Test 95% CI": f"{ci[0]:.2f} – {ci[1]:.2f}" if ci[0] is not None else "—",
                     "Gap": v["gap"], "Verdict": v["verdict"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Train": st.column_config.NumberColumn(format="%.3f"),
                                "Validation": st.column_config.NumberColumn(format="%.3f"),
                                "Test": st.column_config.NumberColumn(format="%.3f"),
                                "Gap": st.column_config.NumberColumn("Gap (train − held-out)", format="%+.3f")})
    st.markdown(" &nbsp; ".join(f"{r['Metric']}: {badge(r['Verdict'], verdict_kind(r['Verdict']))}" for r in rows),
                unsafe_allow_html=True)
    n_sound_test = bt["n_sound_patches"]
    callout("<b>How to read this:</b> overfitting means a model does much better on the images it trained on than on "
            "new ones. The gap is the train score minus the average of validation and test: "
            "below 0.05 = no meaningful overfitting · 0.05–0.10 = mild · above 0.10 = overfitting.<br/>"
            "The <b>test 95% CI</b> comes from resampling whole test images 1,000 times. It is wide because the test "
            f"split holds only {sizes['test']} images and just <b>{n_sound_test} clearly sound tiles</b>, so one tile "
            f"changes balanced accuracy by about {50 / max(n_sound_test, 1):.1f} points. When the train score lies inside "
            "the test interval, the difference is within what chance alone would produce.", "neutral")

    hist = (summary or {}).get("unet", {}).get("history")
    if hist and hist.get("epoch"):
        st.markdown("##### Model B learning curves (full, un-augmented images, every epoch)")
        best_epoch = summary["unet"].get("best_epoch")
        c1, c2 = st.columns(2)
        for col, key, title, rng in [(c1, "loss", "Loss (lower is better)", None),
                                     (c2, "dice", "Dice (higher is better)", [0, 1])]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[f"train_{key}"], mode="lines+markers", name="Train",
                                     line=dict(color="#7f8c8d", width=2)))
            fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[f"val_{key}"], mode="lines+markers", name="Validation",
                                     line=dict(color="#2e86de", width=3)))
            if best_epoch:
                fig.add_vline(x=best_epoch, line_dash="dash", line_color="#27ae60",
                              annotation_text=f"kept: epoch {best_epoch}", annotation_position="top")
            fig.update_layout(title=title, xaxis_title="Epoch", height=330, margin=dict(l=10, r=10, t=50, b=10),
                              yaxis_range=rng, legend=dict(orientation="h", y=-0.25))
            col.plotly_chart(fig, width="stretch")
        u = summary["unet"]
        stop_txt = ("Training stopped early because validation loss stopped improving" if u.get("stopped_early")
                    else "Training ran the full schedule")
        callout(f"{stop_txt}; the weights from <b>epoch {best_epoch}</b> (lowest validation loss) were kept. "
                f"Network size: {u.get('n_params', 0) / 1e6:.1f}M parameters. If the model were memorising, the grey "
                "(train) Dice would keep rising while the blue (validation) Dice fell — the curves moving together "
                "shows it is learning features that generalise.", "neutral")

    cv = (summary or {}).get("baseline", {}).get("cv")
    if cv:
        with st.expander("Model A hyperparameter selection (scene-grouped cross-validation on the training split)"):
            pca_label = lambda v: "none" if v is None or pd.isna(v) else str(int(v))
            chosen_key = (cv["chosen"]["C"], pca_label(cv["chosen"]["pca_components"]))
            df_cv = pd.DataFrame(cv["grid"])
            df_cv["PCA components"] = df_cv["pca_components"].map(pca_label)
            df_cv = df_cv.sort_values(["PCA components", "C"])
            df_cv["CV balanced acc"] = df_cv.apply(lambda r: f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}", axis=1)
            df_cv["Train balanced acc"] = df_cv["train_mean"].map(lambda v: f"{v:.3f}")
            df_cv["Chosen"] = df_cv.apply(lambda r: "✅" if (r["C"], r["PCA components"]) == chosen_key else "", axis=1)
            st.dataframe(df_cv[["PCA components", "C", "CV balanced acc", "Train balanced acc", "Chosen"]],
                         hide_index=True, width="stretch")
            st.markdown(f"Selection rule: the **simplest** setting (no / fewest PCA components, smallest C = strongest "
                        f"regularisation) whose CV score is within one standard error of the best. "
                        f"Chosen: C = {chosen_key[0]}, PCA = {chosen_key[1]}. Larger C values fit the training scenes "
                        f"more closely without improving the cross-validated score, so they are not used.")

# ---------------- 2. Head-to-head on test tiles ----------------
with tab_cmp:
    cmp = pd.DataFrame([
        {"Metric": "Balanced accuracy", "Model A (SVM)": bt["balanced_accuracy"], "Model B (U-Net)": ut["balanced_accuracy"]},
        {"Metric": "Accuracy", "Model A (SVM)": bt["accuracy"], "Model B (U-Net)": ut["accuracy"]},
        {"Metric": "Precision (defect)", "Model A (SVM)": bt["precision"], "Model B (U-Net)": ut["precision"]},
        {"Metric": "Recall (defect)", "Model A (SVM)": bt["recall"], "Model B (U-Net)": ut["recall"]},
        {"Metric": "F1 (defect)", "Model A (SVM)": bt["f1_score"], "Model B (U-Net)": ut["f1_score"]},
        {"Metric": "ROC AUC", "Model A (SVM)": bt["roc_auc"], "Model B (U-Net)": ut["roc_auc"]},
    ])
    c_tab, c_bar = st.columns([1, 1])
    with c_tab:
        st.dataframe(cmp, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Model A (SVM)", "Model B (U-Net)"]})
        st.caption(f"{bt['n_patches']} test tiles: {bt['n_defect_patches']} defect, {bt['n_sound_patches']} sound. "
                   "Balanced accuracy averages the hit rate on defect tiles and on sound tiles, so it is not inflated "
                   "by the class imbalance.")
    with c_bar:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cmp["Metric"], y=cmp["Model A (SVM)"], name="Model A (SVM)", marker_color="#3498db"))
        fig.add_trace(go.Bar(x=cmp["Metric"], y=cmp["Model B (U-Net)"], name="Model B (U-Net)", marker_color="#27ae60"))
        fig.update_layout(barmode="group", yaxis_range=[0, 1.05], height=330, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    for col, m, name, scale in [(c1, bt, "Model A (SVM)", "Blues"), (c2, ut, "Model B (U-Net)", "Greens")]:
        cm = m["confusion_matrix"]
        z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted sound", "Predicted defect"], y=["True sound", "True defect"],
                                   colorscale=scale, showscale=False, text=z, texttemplate="%{text}",
                                   textfont=dict(size=20)))
        fig.update_layout(title=f"{name}: test-tile confusion matrix", height=300, margin=dict(l=10, r=10, t=50, b=10),
                          yaxis=dict(autorange="reversed"))
        col.plotly_chart(fig, width="stretch")
        col.caption(f"Missed defect tiles (FN): {cm['FN']} · False alarms (FP): {cm['FP']}")

# ---------------- 3. Segmentation ----------------
with tab_seg:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Test mean Dice", f"{unet['test']['mean_dice']:.3f}")
    s2.metric("Test mean IoU", f"{unet['test']['mean_iou']:.3f}")
    s3.metric("Pixel recall", f"{unet['test']['pixel_recall'] * 100:.0f}%", help="Share of true corrosion pixels found")
    s4.metric("Pixel precision", f"{unet['test']['pixel_precision'] * 100:.0f}%", help="Share of flagged pixels that are corrosion")
    df_img = pd.DataFrame(unet["test"]["per_image"]).sort_values("dice")
    fig = go.Figure(go.Bar(x=df_img["image_id"].str.replace("mavecodd_", ""), y=df_img["dice"],
                           marker_color=["#e74c3c" if d < 0.5 else ("#f39c12" if d < 0.7 else "#27ae60") for d in df_img["dice"]],
                           hovertext=[f"true area {t * 100:.1f}% · predicted {p * 100:.1f}%"
                                      for t, p in zip(df_img["true_defect_area"], df_img["pred_defect_area"])]))
    fig.update_layout(title="Dice per test image (red < 0.5 · amber 0.5–0.7 · green ≥ 0.7)", yaxis_range=[0, 1],
                      height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    callout("Images with very little corrosion (a few small spots) get low Dice even when the model is reasonable: "
            "a handful of extra or missing pixels is a large fraction of a tiny area. Open those images on the "
            "<b>Image inspection</b> page to see the error map.", "neutral")

# ---------------- 4. Turbidity + cross-domain ----------------
with tab_rob:
    deg = metrics.get("optical_degradation_stress_test") or []
    if deg:
        st.markdown("##### Underwater degradation (test split)")
        df_deg = pd.DataFrame(deg)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["unet_mean_dice"], mode="lines+markers",
                                 name="U-Net Dice", line=dict(color="#8e44ad", width=3)))
        fig.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["unet_patch_balanced_accuracy"], mode="lines+markers",
                                 name="U-Net tile balanced acc", line=dict(color="#27ae60", width=3)))
        if df_deg["baseline_patch_balanced_accuracy"].notna().all():
            fig.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["baseline_patch_balanced_accuracy"],
                                     mode="lines+markers", name="SVM tile balanced acc", line=dict(color="#3498db", width=3)))
        fig.update_layout(xaxis_title="Simulated turbidity (0 = clear water)", yaxis_range=[0, 1.05], height=360,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Test images are artificially degraded (red-light absorption, haze, blur, marine snow) with fixed "
                   "random seeds. Training used mild random degradation as augmentation.")

    cross = metrics.get("cross_domain_generalization")
    st.markdown("##### Cross-domain: dry-dock corrosion model on underwater cracks")
    if cross:
        x1, x2, x3 = st.columns(3)
        x1.metric("Dice", f"{cross['cross_domain_dice']:.3f}")
        x2.metric("IoU", f"{cross['cross_domain_iou']:.3f}")
        x3.metric("Pixel recall", f"{cross['cross_domain_pixel_recall'] * 100:.1f}%")
        callout(f"Zero-shot on {cross['n_images']} Underwater Crack test images. This is <b>expected to be near zero</b>: "
                "the model learned what rust looks like in open-air dry-dock photos. Thin dark cracks on submerged "
                "concrete differ in both appearance and shape. Transfer would need labelled target-domain data "
                "(fine-tuning or domain adaptation).", "neutral")
    else:
        st.info("Cross-domain test not run (Underwater Crack dataset missing).")

# ---------------- 5. Safety ----------------
with tab_safe:
    callout(f"""
<h4>Why missed defects matter most</h4>
<ul>
<li><b>False positive</b> (false alarm): the ROV re-surveys a clean plate or a diver takes a second look — a small cost.</li>
<li><b>False negative</b> (missed defect): corrosion goes unreported and can progress. On the test tiles,
Model A missed <b>{bt['false_negatives']}</b> of {bt['n_defect_patches']} defect tiles and Model B missed
<b>{ut['false_negatives']}</b>.</li>
</ul>
Neither model is perfect; both are screening aids whose alarms must be confirmed by a surveyor.
""", "danger")
    callout("""
<h4>Known limitations</h4>
<ul>
<li><b>Small test set:</b> 16 images and 21 clearly sound tiles, so test scores have wide confidence intervals.</li>
<li><b>U-Net false alarms:</b> mostly deck railings / sky at the top of dry-dock photos and weathered coating that the
annotators did not label.</li>
<li><b>Dry-dock, not underwater:</b> MaVeCoDD photos are taken in air; underwater conditions are simulated.</li>
<li><b>Surface only:</b> a camera cannot measure plate thinning or internal fatigue (that needs ultrasonic testing).</li>
</ul>
""", "neutral")
