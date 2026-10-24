"""Streamlit Page 2: Inspect one image with both models and compare against ground truth."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Image Inspection | Hull AI", layout="wide")

from src.utils.reproducibility import get_device
from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask
from src.data.preprocessing import apply_underwater_degradation, resize_long_side
from src.data.split_dataset import load_split_manifest
from src.models.cnn import compute_segmentation_metrics
from src.models.evaluate import unet_patch_scores
from src.models.predict import HullInspectionPredictor
from app.ui import inject_css, callout, badge, legend_chip, dice_quality, load_metrics, page_header

inject_css()
page_header("🔬 Image inspection",
            "Run both models on one photo and compare the result with the human-labelled ground truth.")

checkpoints_dir = ROOT / "models" / "checkpoints"
baseline_path = checkpoints_dir / "baseline_mavecodd.joblib"
unet_path = checkpoints_dir / "unet_mavecodd.pth"
if not unet_path.exists():
    callout("Model checkpoints not found. Run <b>python scripts/train_model.py --dataset mavecodd</b> first.", "danger")
    st.stop()


@st.cache_resource
def load_predictor(mtime: float):
    return HullInspectionPredictor(
        baseline_path=baseline_path if baseline_path.exists() else None,
        unet_path=unet_path,
        device=get_device("auto"),
    )


predictor = load_predictor(unet_path.stat().st_mtime)
metrics = load_metrics("mavecodd")
typical_dice = metrics["deep_learning_model_b"]["test"]["mean_dice"] if metrics else None

SPLIT_TAGS = {
    "test": "TEST · unseen",
    "val": "VAL · used for early stopping & thresholds",
    "train": "TRAIN · seen in training",
}


@st.cache_data(show_spinner=False)
def list_samples(source: str):
    if source == "mavecodd":
        manifest = ROOT / "data/processed/split_manifest_mavecodd.json"
        if manifest.exists():
            splits = load_split_manifest(manifest)
            out = []
            for split in ("test", "val", "train"):
                for s in sorted(splits[split], key=lambda x: x["image_id"]):
                    out.append({**s, "split": split})
            return out
        return [{**s, "split": "unknown"} for s in
                scan_mavecodd_samples(ROOT / "data/raw/mavecodd", zip_fallback=ROOT / "ry392rp8cj-1.zip")]
    if source == "crack":
        manifest = ROOT / "data/processed/split_manifest_crack.json"
        if manifest.exists():
            return [{**s, "split": "test"} for s in load_split_manifest(manifest)["test"][:60]]
        return [{**s, "split": "unknown"} for s in
                scan_crack_samples(ROOT / "data/raw/underwater_crack", zip_fallback=ROOT / "under-crack-images.zip", limit=60)]
    sample_dir = ROOT / "data/sample/images"
    return [{"image_path": str(f), "mask_path": str(ROOT / "data/sample/masks" / f"{f.stem}_mask.png"),
             "image_id": f.stem, "dataset_source": "synthetic", "split": "unseen"}
            for f in sorted(sample_dir.glob("*.jpg"))] if sample_dir.exists() else []


@st.cache_data(show_spinner=False, max_entries=8)
def load_sample(sample: dict):
    img, mask = load_image_and_mask(sample)
    # Work at <= 1600 px: plenty for display, much faster than 12 MP originals
    if max(img.shape[:2]) > 1600:
        img, mask = resize_long_side(img, mask, 1600)
    return img, mask


def gt_overlay(img, mask):
    vis = img.copy()
    m = mask > 0
    vis[m] = (0.45 * vis[m] + 0.55 * np.array([255, 200, 0])).astype(np.uint8)
    return vis


# ---------------- Input selection ----------------
SOURCES = {
    "MaVeCoDD": "mavecodd",
    "Cracks": "crack",
    "Synthetic": "synthetic",
    "Upload": "upload",
}
image, gt_mask, chosen, split = None, None, None, None
defect_hint = "corrosion"

with st.container(border=True):
    c_src, c_pick = st.columns([2, 3])
    with c_src:
        source_label = st.segmented_control("Image source", list(SOURCES), default="MaVeCoDD",
                                            help="MaVeCoDD hull corrosion is what the models were trained on. "
                                                 "Underwater cracks and synthetic plates show the models' limits.")
        source = SOURCES.get(source_label or "MaVeCoDD")
        split_filter = None
        if source == "mavecodd":
            split_filter = st.segmented_control(
                "Show images from", ["Test (unseen)", "Validation", "Training", "All"], default="Test (unseen)",
                help="Judge the model on TEST images: neither the model nor its thresholds ever saw them.")
    with c_pick:
        if source == "upload":
            uploaded = st.file_uploader("Upload a hull photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                image = np.array(Image.open(uploaded).convert("RGB"))
                if max(image.shape[:2]) > 1600:
                    image, _ = resize_long_side(image, None, 1600)
                split = "upload"
        else:
            samples = list_samples(source)
            wanted = {"Test (unseen)": "test", "Validation": "val", "Training": "train"}.get(split_filter or "All")
            if wanted:
                samples = [s for s in samples if s["split"] == wanted]
            if not samples:
                callout("No images found for this source. Run <b>python scripts/prepare_dataset.py</b>.", "warn")
            else:
                def fmt(i):
                    s = samples[i]
                    tag = SPLIT_TAGS.get(s["split"], s["split"].upper())
                    tier = f" ({s['resolution_tier']})" if s.get("resolution_tier") else ""
                    return f"[{tag}] {s['image_id'].replace('mavecodd_', '')}{tier}"
                idx = st.selectbox(f"Select image ({len(samples)} available)", range(len(samples)), format_func=fmt)
                chosen = samples[idx]
                split = chosen["split"]
                defect_hint = "crack" if source == "crack" else "corrosion"
                with st.spinner("Loading image..."):
                    image, gt_mask = load_sample(chosen)

        with st.expander("🌊 Simulate underwater conditions"):
            enable_degradation = st.toggle("Apply underwater degradation", value=False)
            d1, d2, d3 = st.columns(3)
            turbidity = d1.slider("Turbidity", 0.0, 0.8, 0.25, 0.05, disabled=not enable_degradation)
            red_att = d2.slider("Red absorption", 0.0, 0.9, 0.65, 0.05, disabled=not enable_degradation)
            contrast = d3.slider("Contrast kept", 0.2, 1.0, 0.80, 0.05, disabled=not enable_degradation)

# Context: what this result can and cannot tell you
if split == "train":
    callout("<b>This image was used to train the models.</b> A good result here does not prove the model "
            "works on new photos. Choose <b>Test (unseen)</b> to judge real performance.", "warn")
elif split == "val":
    callout("This image was <b>not</b> used to fit the model weights, but it was used to decide when to stop "
            "training and to pick the decision thresholds. TEST images are the fully independent check.", "neutral")
elif split == "test" and source == "mavecodd":
    callout("<b>Held-out test image</b> — the models never saw this photo (or any frame from the same scene) "
            "during training or tuning. This is an honest check of real performance.", "success")
if source == "crack":
    callout("<b>Expected to perform poorly.</b> The models were trained only on dry-dock <i>corrosion</i>. "
            "Thin underwater <i>cracks</i> are a different defect in a different environment; this source "
            "demonstrates the cross-domain limitation, not a bug.", "warn")
if source == "synthetic":
    callout("Synthetic cartoon-like plates were <b>not</b> used to train the MaVeCoDD models, so results "
            "here are not representative of real performance.", "neutral")
if split == "upload":
    callout("No ground truth exists for an uploaded photo, so the prediction can be shown but not scored.", "neutral")

if image is None:
    st.stop()

proc_image = image
if enable_degradation:
    proc_image = apply_underwater_degradation(image, turbidity=turbidity, red_attenuation=red_att,
                                              contrast_factor=contrast, random_seed=0)

with st.spinner("Running Model A (SVM) and Model B (U-Net)..."):
    res = predictor.predict(proc_image, defect_hint=defect_hint, gt_mask=gt_mask)

has_gt = gt_mask is not None
base = res.get("baseline_result")

# Baseline tile scoring (used in the summary and in its tab)
tile_correct = tile_scored = 0
if base and has_gt:
    for t in base["tiles"]:
        if t["true_label"] >= 0:
            tile_scored += 1
            tile_correct += int(t["prediction"] == t["true_label"])

# ---------------- Summary strip ----------------
with st.container(border=True):
    if has_gt:
        m = compute_segmentation_metrics(res["probability_map"], gt_mask, threshold=predictor.mask_threshold)
        label, kind = dice_quality(m["dice"])
        s_left, s1, s2, s3, s4 = st.columns([2.2, 1, 1, 1, 1])
        with s_left:
            st.markdown(f"#### Verdict: {badge(label, kind)}", unsafe_allow_html=True)
            ref = f" Average on all test images: {typical_dice:.2f}." if typical_dice is not None else ""
            st.caption(f"U-Net flagged {res['defect_area_fraction'] * 100:.1f}% of the image; the human label marks "
                       f"{gt_mask.mean() * 100:.1f}%.{ref}")
        s1.metric("Dice", f"{m['dice']:.2f}", help="Overall overlap: 2×overlap / (predicted + true area). 1.0 = perfect.")
        s2.metric("Recall", f"{m['recall'] * 100:.0f}%", help="Corrosion found: share of the true corrosion pixels the model flagged.")
        s3.metric("Precision", f"{m['precision'] * 100:.0f}%", help="Correct alarms: share of flagged pixels that really are corrosion.")
        s4.metric("SVM tiles", f"{tile_correct}/{tile_scored}" if tile_scored else "—",
                  help="Model A: correctly classified tiles out of the clearly-labelled 256 px tiles of this image.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Decision", "Suspected defect" if res["defect_detected"] else "Sound hull")
        s2.metric("Area flagged", f"{res['defect_area_fraction'] * 100:.1f}%")
        s3.metric("Confidence", f"{res['confidence_score'] * 100:.0f}%",
                  help="If flagged: mean defect probability inside the flagged region. If not: 1 − mean of the "
                       "1% most suspicious pixels.")

# ---------------- Detail tabs ----------------
tab_names = ["🧠 U-Net vs ground truth"] + (["🎯 Error map"] if has_gt else []) + \
            (["🧮 SVM tiles"] if base else []) + ["🌡️ Probability heatmap", "📍 Defect regions"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs["🧠 U-Net vs ground truth"]:
    cols = st.columns(3 if has_gt else 2)
    cols[0].image(proc_image, caption=f"Input ({image.shape[1]}×{image.shape[0]})", width="stretch")
    if has_gt:
        cols[1].image(gt_overlay(proc_image, gt_mask),
                      caption=f"Ground truth (yellow) · {gt_mask.mean() * 100:.1f}% corrosion", width="stretch")
    cols[-1].image(predictor.draw_overlay(proc_image, res),
                   caption=f"U-Net prediction (magenta) · {res['defect_area_fraction'] * 100:.1f}% flagged",
                   width="stretch")
    st.caption(f"Pixel threshold {predictor.mask_threshold} · frame alarm when ≥ "
               f"{predictor.detection_area_threshold * 100:.0f}% of the area is flagged (both chosen on the "
               f"validation split) · inference {res['inference_time_ms']:.0f} ms on {predictor.device}.")

if has_gt:
    with tabs["🎯 Error map"]:
        c_err, c_txt = st.columns([3, 2])
        with c_err:
            st.image(predictor.draw_error_map(proc_image, res["binary_mask"], gt_mask), width="stretch",
                     caption="Prediction compared with ground truth, pixel by pixel")
        with c_txt:
            st.markdown(legend_chip("#00c846", "found correctly") + "<br/>" + legend_chip("#eb2828", "false alarm")
                        + "<br/>" + legend_chip("#286eff", "missed"), unsafe_allow_html=True)
            st.write("")
            callout(
                f"The model found <b>{m['recall'] * 100:.0f}%</b> of the real corrosion, and "
                f"<b>{m['precision'] * 100:.0f}%</b> of what it flagged really is corrosion (IoU {m['iou']:.2f})."
                "<br/><br/>Rule of thumb: Dice ≥ 0.70 good · 0.50–0.70 fair · below 0.50 poor. Corrosion edges "
                "are fuzzy, so even two human annotators rarely agree above ~0.8.", "neutral")

if base:
    with tabs["🧮 SVM tiles"]:
        st.markdown("The baseline cannot draw a mask; it labels each 256 px tile (of the image rescaled to "
                    "1024 px) as **defect** or **sound**. Tiles are scored only when they are clearly one or the "
                    "other (≥ 5% or ≤ 0.5% corrosion).")
        vis = proc_image.copy()
        thick = max(2, int(max(vis.shape[:2]) / 350))
        for t in base["tiles"]:
            if has_gt and t["true_label"] >= 0:
                color = (0, 200, 70) if t["prediction"] == t["true_label"] else (235, 40, 40)
            else:
                color = (170, 170, 170)
            cv2.rectangle(vis, (t["x1"] + thick, t["y1"] + thick), (t["x2"] - thick, t["y2"] - thick), color, thick)
            tag = "D" if t["prediction"] else "S"
            fs = max(0.6, vis.shape[1] / 900)
            pos = (t["x1"] + 4 * thick, t["y1"] + int(28 * fs))
            cv2.putText(vis, tag, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick + 2)
            cv2.putText(vis, tag, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)
        c_a, c_b = st.columns([3, 2])
        c_a.image(vis, width="stretch", caption="Letters = SVM prediction (D = defect, S = sound)")
        with c_b:
            st.markdown(legend_chip("#00c846", "correct") + "<br/>" + legend_chip("#eb2828", "wrong") + "<br/>"
                        + legend_chip("#aaaaaa", "not scored (ambiguous / no ground truth)"), unsafe_allow_html=True)
            st.metric("Tiles predicted as defect", f"{base['defect_tiles']} / {base['total_tiles']}")
            if tile_scored:
                st.metric("SVM accuracy on this image", f"{tile_correct / tile_scored * 100:.0f}%  ({tile_correct}/{tile_scored})")
                yt, fr = unet_patch_scores(res["probability_map"], gt_mask, predictor.mask_threshold,
                                           **{**predictor.patch_config, "defect_min_frac": 0.05, "sound_max_frac": 0.005})
                if yt:
                    u_acc = np.mean((np.array(fr) >= predictor.detection_area_threshold).astype(int) == np.array(yt))
                    st.metric("U-Net accuracy on the same tiles", f"{u_acc * 100:.0f}%")
            st.metric("SVM latency", f"{base['latency_ms']:.0f} ms")

with tabs["🌡️ Probability heatmap"]:
    pm = res["probability_map"]
    if max(pm.shape) > 800:
        pm = cv2.resize(pm, (800, int(800 * pm.shape[0] / pm.shape[1])), interpolation=cv2.INTER_AREA)
    fig = px.imshow(pm, color_continuous_scale="Turbo", zmin=0, zmax=1)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("Per-pixel corrosion probability from the U-Net (0 = sound, 1 = corrosion).")

with tabs["📍 Defect regions"]:
    if res["bounding_boxes"]:
        st.dataframe([{
            "ID": b["defect_id"], "BBox [x1, y1, x2, y2]": str(b["bbox_pixel"]),
            "Centroid (norm.)": f"({b['centroid_normalized'][0]:.2f}, {b['centroid_normalized'][1]:.2f})",
            "Area": f"{b['area_fraction'] * 100:.2f}%",
        } for b in res["bounding_boxes"]], width="stretch", hide_index=True)
    else:
        st.info("No connected defect regions above the minimum size.")

callout("<b>Formal notice:</b> predictions are suspected surface defects requiring on-site confirmation. "
        "They do not measure plate thickness or structural stress.", "warn")
