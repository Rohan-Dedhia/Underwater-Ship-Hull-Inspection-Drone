"""Shared dashboard helpers: theme-safe styling and metric loading.

Every custom HTML block sets BOTH its background and its text colour explicitly, so it stays
readable whether Streamlit runs in light or dark theme (inheriting the theme's text colour
on a hard-coded light background is what caused white-on-white text before).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

_CSS = """
<style>
.callout {border-radius: 8px; padding: 14px 18px; margin: 6px 0 14px 0; font-size: 0.93rem; line-height: 1.55;}
.callout, .callout * {color: inherit !important;}
.callout h4 {margin: 0 0 6px 0; font-size: 1.02rem;}
.callout ul {margin: 4px 0 0 0; padding-left: 20px;}
.callout.info    {background: #e8f1fb; color: #0b3a66 !important; border-left: 5px solid #1f77d0;}
.callout.warn    {background: #fff4e0; color: #5c3b00 !important; border-left: 5px solid #e69500;}
.callout.danger  {background: #fdecec; color: #6b1111 !important; border-left: 5px solid #d93025;}
.callout.success {background: #e7f6ec; color: #0f4d26 !important; border-left: 5px solid #1e9e4a;}
.callout.neutral {background: #f1f3f5; color: #1f2933 !important; border-left: 5px solid #7b8794;}

.kpi-card {background-color: #1b3346; background-image: linear-gradient(135deg, #1e3d59 0%, #17252a 100%); border-radius: 12px;
           padding: 18px 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
           min-height: 118px; display: flex; flex-direction: column; justify-content: center;}
.kpi-card .kpi-value {font-size: 1.85rem; font-weight: 700; color: #4deeea !important;}
.kpi-card .kpi-label {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;
                      color: #cfd8dc !important; margin-top: 4px;}

.badge {display: inline-block; padding: 3px 12px; border-radius: 14px; font-weight: 600; font-size: 0.85rem;}
.badge.good    {background: #e7f6ec; color: #0f4d26 !important; border: 1px solid #1e9e4a;}
.badge.fair    {background: #fff4e0; color: #5c3b00 !important; border: 1px solid #e69500;}
.badge.poor    {background: #fdecec; color: #6b1111 !important; border: 1px solid #d93025;}
.badge.neutral {background: #f1f3f5; color: #1f2933 !important; border: 1px solid #7b8794;}

.legend-chip {display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin: 0 6px -2px 12px;}
.hud-status {background: #17252a; color: #4deeea !important; padding: 10px; border-radius: 6px;
             font-family: monospace; font-size: 0.85rem;}
.hud-status * {color: #4deeea !important;}

/* Page header banner: dark gradient with explicit light text (readable in both themes) */
.page-hero {background-color: #0e4a5c; background-image: linear-gradient(120deg, #0b2a3c 0%, #0e4a5c 55%, #127a7a 100%);
            border-radius: 14px; padding: 22px 26px; margin: 0 0 18px 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);}
.page-hero .hero-title {font-size: 1.85rem; font-weight: 700; color: #ffffff !important; line-height: 1.25;}
.page-hero .hero-sub {font-size: 0.98rem; color: #cfe8ec !important; margin-top: 6px;}
.page-hero .hero-chip {display: inline-block; margin: 10px 8px 0 0; padding: 3px 11px; border-radius: 12px;
                       font-size: 0.78rem; font-weight: 600; background: rgba(255,255,255,0.14);
                       color: #e8f7f9 !important; border: 1px solid rgba(255,255,255,0.25);}

/* Metric tiles as subtle cards; translucent colours adapt to light and dark themes */
[data-testid="stMetric"] {background: rgba(127,127,127,0.07); border: 1px solid rgba(127,127,127,0.22);
                          border-radius: 10px; padding: 10px 14px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}

/* Tabs: larger, bolder labels */
.stTabs [data-baseweb="tab"] p {font-size: 0.98rem; font-weight: 600;}

/* Step number bubble used on the home page */
.step-num {display: inline-block; width: 28px; height: 28px; border-radius: 50%; text-align: center;
           line-height: 28px; font-weight: 700; background: #127a7a; color: #ffffff !important; margin-right: 8px;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def callout(html: str, kind: str = "info") -> None:
    """Coloured message box with explicit text colour. `html` may contain simple HTML tags."""
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", chips: Optional[list] = None) -> None:
    """Consistent page banner. `chips` are short tags shown under the subtitle."""
    chip_html = "".join(f'<span class="hero-chip">{c}</span>' for c in (chips or []))
    st.markdown(f'<div class="page-hero"><div class="hero-title">{title}</div>'
                f'<div class="hero-sub">{subtitle}</div>{chip_html}</div>', unsafe_allow_html=True)


def step_number(n: int) -> str:
    return f'<span class="step-num">{n}</span>'


def sidebar_status() -> None:
    """Small always-visible panel: are the models trained, and how good are they on test data?"""
    ckpt = ROOT / "models" / "checkpoints"
    unet_ok = (ckpt / "unet_mavecodd.pth").exists()
    svm_ok = (ckpt / "baseline_mavecodd.joblib").exists()
    metrics = load_metrics("mavecodd")
    with st.sidebar.container(border=True):
        st.markdown("**System status**")
        st.markdown(f"{'🟢' if unet_ok else '🔴'} U-Net segmenter  \n"
                    f"{'🟢' if svm_ok else '🔴'} SVM tile classifier")
        if metrics:
            u = metrics["deep_learning_model_b"]["test"]
            st.caption(f"Test Dice {u['mean_dice']:.2f} · missed defect tiles "
                       f"{u['patch_classification']['false_negatives']}/{u['patch_classification']['n_defect_patches']}")
        else:
            st.caption("Run training + evaluation to see results.")


def kpi_card(value: str, label: str) -> None:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{text}</span>'


def legend_chip(color: str, label: str) -> str:
    return f'<span class="legend-chip" style="background:{color};"></span>{label}'


def dice_quality(dice: float) -> Tuple[str, str]:
    """Plain-language rating of a segmentation Dice score."""
    if dice >= 0.70:
        return "Good match", "good"
    if dice >= 0.50:
        return "Fair match", "fair"
    return "Poor match", "poor"


def verdict_kind(verdict: str) -> str:
    return {"No meaningful overfitting": "good", "Mild overfitting": "fair"}.get(verdict, "poor")


def load_metrics(dataset: str = "mavecodd") -> Optional[Dict[str, Any]]:
    """Evaluation metrics written by scripts/evaluate_model.py (None if missing or outdated)."""
    f = ROOT / "outputs" / "metrics" / f"test_metrics_{dataset}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return data if "overfitting_check" in data else None


def load_training_summary(dataset: str = "mavecodd") -> Optional[Dict[str, Any]]:
    f = ROOT / "outputs" / "metrics" / f"training_summary_{dataset}.json"
    return json.loads(f.read_text()) if f.exists() else None


def load_split_lookup(dataset: str = "mavecodd") -> Dict[str, str]:
    """image_id -> 'train' / 'val' / 'test'."""
    f = ROOT / "data" / "processed" / f"split_manifest_{dataset}.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    return {s["image_id"]: split for split, v in data.items() for s in v["samples"]}


def missing_metrics_message() -> None:
    callout("<b>Evaluation results not found or outdated.</b> Run "
            "<b>python scripts/train_model.py --dataset mavecodd</b> and then "
            "<b>python scripts/evaluate_model.py --dataset mavecodd</b>.", "warn")
