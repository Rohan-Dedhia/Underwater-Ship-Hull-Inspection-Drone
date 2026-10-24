import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Dataset Explorer | Hull AI", layout="wide")

from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask
from src.data.split_dataset import load_split_manifest

from app.ui import inject_css, page_header, badge, load_split_lookup

inject_css()
page_header("🗂️ Dataset explorer", "Browse the photos and their human-drawn corrosion masks, and see exactly how the data was split.")

# Dataset selection
ds_tab = st.segmented_control("Dataset", ["MaVeCoDD (Marine Vessel Corrosion)", "Underwater Crack Dataset", "Train/Val/Test Splits Manifest"], default="MaVeCoDD (Marine Vessel Corrosion)") or "MaVeCoDD (Marine Vessel Corrosion)"

if ds_tab == "MaVeCoDD (Marine Vessel Corrosion)":
    st.subheader("🚢 MaVeCoDD: Marine Vessel Corrosion in Dry-Dock")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/ry392rp8cj.1`), Kamzelis et al.
    - **Subject**: Real vessel hulls undergoing dry-dock repair and blasting.
    - **Annotation Type**: Pixel-level segmentation masks (`*_label.jpg`). Defect pixels retain color; background is near-zero black.
    - **Resolution Tiers**: HiRes (up to 4032 &times; 3024) and LoRes (1920 &times; 1080).
    """)

    mav_samples = scan_mavecodd_samples(ROOT / "data/raw/mavecodd", zip_fallback=ROOT / "ry392rp8cj-1.zip")
    st.metric("Total Valid Paired Samples", len(mav_samples))

    if mav_samples:
        df_mav = pd.DataFrame(mav_samples)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("##### Resolution Distribution")
            fig_pie = px.pie(df_mav, names="resolution_tier", title="MaVeCoDD Resolution Split", color_discrete_sequence=["#0984e3", "#00cec9"])
            st.plotly_chart(fig_pie, width="stretch")

        with c2:
            st.write("##### Sample Gallery & Mask Inspector")
            sample_idx = st.slider("Select Sample Index", 0, len(mav_samples)-1, 0)
            s = mav_samples[sample_idx]
            img, mask = load_image_and_mask(s)
            split_of = load_split_lookup("mavecodd").get(s["image_id"])
            kind = {"test": "good", "val": "fair", "train": "neutral"}.get(split_of, "neutral")
            st.markdown(f"**{s['image_id'].replace('mavecodd_', '')}** · scene group `{s['scene_id']}` · "
                        f"split {badge((split_of or 'unknown').upper(), kind)}", unsafe_allow_html=True)

            alpha = st.slider("Mask Overlay Opacity", 0.0, 1.0, 0.4, 0.05)
            color_mask = np.zeros_like(img)
            color_mask[mask > 0] = [231, 76, 60]  # Red overlay
            blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

            c_img1, c_img2 = st.columns(2)
            c_img1.image(img, caption=f"Raw Image: {s['image_id']} ({img.shape[1]}x{img.shape[0]})", width="stretch")
            c_img2.image(blended, caption=f"Ground-Truth Mask (Defect: {(np.sum(mask>0)/mask.size)*100:.2f}% of pixels)", width="stretch")

elif ds_tab == "Underwater Crack Dataset":
    st.subheader("🌊 Underwater Concrete & Submerged Structure Cracks")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/g962rbm77b/1`), Gao et al.
    - **Subject**: Submerged underwater structures and concrete piers with visible fissure cracking.
    - **Annotation Type**: Binary PNG segmentation masks (Crack = 255, Background = 0).
    - **Standard Dimensions**: 512 &times; 512 &times; 3 RGB.
    """)

    crack_samples = scan_crack_samples(ROOT / "data/raw/underwater_crack", zip_fallback=ROOT / "under-crack-images.zip", limit=100)
    st.metric("Indexed Samples (Showing top 100 of 1,037)", len(crack_samples))

    if crack_samples:
        sample_idx = st.slider("Select Crack Sample Index", 0, len(crack_samples)-1, 0)
        s = crack_samples[sample_idx]
        img, mask = load_image_and_mask(s)

        alpha = st.slider("Crack Mask Overlay Opacity", 0.0, 1.0, 0.45, 0.05)
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = [255, 107, 107]
        blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Underwater Image: {s['image_id']} (512x512)", width="stretch")
        c2.image(blended, caption=f"Ground-Truth Crack Mask (Coverage: {(np.sum(mask>0)/mask.size)*100:.2f}%)", width="stretch")

else:
    st.subheader("📊 Train / Validation / Test Split")
    st.markdown("""
- **Grouping**: consecutive frame numbers (e.g. `colorImage1005`–`1009`) are near-duplicates from the same video or
  photo burst, so they form one *scene group*. A group is assigned to exactly one split.
- **Stratification**: HiRes and LoRes groups are split separately, so every split contains both camera types.
- **Proportions**: about 60% train / 20% validation / 20% test of the scene groups.
- **Use**: train = fit model weights · validation = early stopping + decision thresholds · test = final numbers only.
    """)

    manifest_file = ROOT / "data" / "processed" / "split_manifest_mavecodd.json"
    if manifest_file.exists():
        splits = load_split_manifest(manifest_file)
        rows = []
        for name in ["train", "val", "test"]:
            items = splits[name]
            rows.append({
                "Split": name,
                "Images": len(items),
                "HiRes": sum(1 for s in items if s.get("resolution_tier") == "HiRes"),
                "LoRes": sum(1 for s in items if s.get("resolution_tier") == "LoRes"),
                "Scene groups": len({s["scene_id"] for s in items}),
                "Groups": ", ".join(sorted({s["scene_id"] for s in items})),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        all_train = {s["scene_id"] for s in splits["train"]}
        overlap = all_train & ({s["scene_id"] for s in splits["val"]} | {s["scene_id"] for s in splits["test"]})
        st.markdown("**Leakage check:** " + ("✅ no scene group appears in more than one split." if not overlap
                                              else f"❌ shared groups: {sorted(overlap)}"))

        with st.expander("Every image and its split"):
            st.dataframe(pd.DataFrame([{"image_id": s["image_id"], "scene group": s["scene_id"], "split": name}
                                       for name in ["train", "val", "test"] for s in splits[name]]),
                         width="stretch", hide_index=True)
    else:
        st.warning("Split manifest not found. Run `python scripts/prepare_dataset.py` to generate.")
