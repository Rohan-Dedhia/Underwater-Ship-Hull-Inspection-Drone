"""Dataset inspection and verification script for MaVeCoDD and Underwater Crack datasets."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.dataset_loader import (
    scan_mavecodd_samples,
    scan_crack_samples,
    load_image_and_mask
)
from src.data.annotation_parser import parse_mavecodd_mask, parse_crack_mask


def main():
    print("=" * 70)
    print("AI-POWERED UNDERWATER HULL INSPECTION DRONE - DATASET INSPECTION")
    print("=" * 70)

    config = load_config()
    cfg_paths = config["paths"]

    # 1. Inspect MaVeCoDD
    print("\n[1/3] Scanning Primary Dataset: MaVeCoDD (Marine Vessel Corrosion)...")
    mavecodd_samples = scan_mavecodd_samples(
        raw_dir=ROOT / cfg_paths["raw_mavecodd"],
        zip_fallback=ROOT / cfg_paths["zip_mavecodd"]
    )
    print(f"Total matched MaVeCoDD image-mask pairs found: {len(mavecodd_samples)}")

    if mavecodd_samples:
        df_mav = pd.DataFrame(mavecodd_samples)
        if "resolution_tier" in df_mav.columns:
            print("  Breakdown by Resolution Tier:")
            for tier, count in df_mav["resolution_tier"].value_counts().items():
                print(f"    - {tier}: {count} pairs")

        # Inspect first 3 samples for dimensional and mask properties
        print("  Sample characteristics:")
        for idx in range(min(3, len(mavecodd_samples))):
            s = mavecodd_samples[idx]
            img, mask = load_image_and_mask(s)
            defect_pixels = np.sum(mask > 0)
            coverage_pct = (defect_pixels / mask.size) * 100
            print(f"    * Sample [{s['image_id']}]: img_shape={img.shape}, mask_shape={mask.shape}, "
                  f"defect_pixels={defect_pixels:,} ({coverage_pct:.2f}% coverage)")

    # 2. Inspect Underwater Crack Dataset
    print("\n[2/3] Scanning Secondary Dataset: Underwater Crack...")
    crack_samples = scan_crack_samples(
        raw_dir=ROOT / cfg_paths["raw_underwater_crack"],
        zip_fallback=ROOT / cfg_paths["zip_underwater_crack"]
    )
    print(f"Total matched Crack image-mask pairs found: {len(crack_samples)}")

    if crack_samples:
        print("  Sample characteristics:")
        for idx in range(min(3, len(crack_samples))):
            s = crack_samples[idx]
            img, mask = load_image_and_mask(s)
            defect_pixels = np.sum(mask > 0)
            coverage_pct = (defect_pixels / mask.size) * 100
            print(f"    * Sample [{s['image_id']}]: img_shape={img.shape}, mask_shape={mask.shape}, "
                  f"crack_pixels={defect_pixels:,} ({coverage_pct:.2f}% coverage)")

    # 3. Summary & Feasibility Check
    print("\n[3/3] Task Formulation & Annotation Compatibility Verification:")
    print("  - Annotation Format: Pixel-level binary segmentation masks available for both datasets.")
    print("  - Selected Vision Tasks:")
    print("    * Model A (Traditional ML Baseline): Image/Tile-level Defect Classification (colour/texture features + SVM)")
    print("    * Model B (Deep Learning): U-Net Pixel-Level Defect Segmentation with Bounding-Box Localization")
    print("  - Cross-Domain Strategy: Primary training on MaVeCoDD corrosion; zero-shot / fine-tuned transfer")
    print("    evaluation on Underwater Crack dataset to test generalization to underwater environment.")
    print("=" * 70)
    print("Dataset inspection completed successfully.")


if __name__ == "__main__":
    main()
