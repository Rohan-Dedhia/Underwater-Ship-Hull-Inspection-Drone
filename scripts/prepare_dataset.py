"""Dataset preparation script: extraction, synthetic sample generation, and stratified splitting."""

import os
import sys
import zipfile
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.data.dataset_loader import (
    scan_mavecodd_samples,
    scan_crack_samples,
    generate_synthetic_samples
)
from src.data.split_dataset import create_leakage_free_splits, save_split_manifest


def main():
    print("=" * 70)
    print("AI-POWERED UNDERWATER HULL INSPECTION DRONE - DATASET PREPARATION")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("random_seed", 42))
    cfg_paths = config["paths"]

    # 1. Extract zip archives if not extracted
    raw_mav_dir = ROOT / cfg_paths["raw_mavecodd"]
    zip_mav = ROOT / cfg_paths["zip_mavecodd"]
    if zip_mav.exists() and not (raw_mav_dir / "HiRes").exists():
        print(f"Extracting {zip_mav.name} to {raw_mav_dir}...")
        raw_mav_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_mav, "r") as z:
            z.extractall(raw_mav_dir)
        print("Extracted MaVeCoDD dataset.")
    elif (raw_mav_dir / "HiRes").exists():
        print(f"MaVeCoDD dataset found at: {raw_mav_dir}")
    else:
        print(f"MaVeCoDD dataset NOT found (expected {zip_mav.name} in repo root or files in {raw_mav_dir}).")

    raw_crack_dir = ROOT / cfg_paths["raw_underwater_crack"]
    zip_crack = ROOT / cfg_paths["zip_underwater_crack"]
    if zip_crack.exists() and not (raw_crack_dir / "images").exists():
        print(f"Extracting {zip_crack.name} to {raw_crack_dir}...")
        raw_crack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_crack, "r") as z:
            z.extractall(raw_crack_dir)
        print("Extracted Underwater Crack dataset.")
    elif (raw_crack_dir / "images").exists():
        print(f"Underwater Crack dataset found at: {raw_crack_dir}")
    else:
        print(f"Underwater Crack dataset NOT found (expected {zip_crack.name} in repo root).")

    # 2. Generate Synthetic Sample Dataset (Quick-start mode)
    sample_dir = ROOT / cfg_paths["sample_dir"]
    print(f"Generating synthetic quick-start samples at {sample_dir}...")
    sample_items = generate_synthetic_samples(
        output_dir=sample_dir,
        num_samples=config["dataset"].get("sample_count", 40),
        image_size=tuple(config["dataset"].get("image_size", [256, 256])),
        random_seed=config.get("random_seed", 42)
    )
    print(f"Generated {len(sample_items)} synthetic hull samples.")

    # 3. Create Split Manifests with Leakage Prevention
    processed_dir = ROOT / cfg_paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 3a. MaVeCoDD splits
    mav_samples = scan_mavecodd_samples(
        raw_mav_dir, zip_fallback=zip_mav,
        scene_block_size=config["dataset"].get("scene_block_size", 5)
    )
    if mav_samples:
        mav_splits = create_leakage_free_splits(
            mav_samples,
            train_ratio=config["dataset"]["train_split"],
            val_ratio=config["dataset"]["val_split"],
            test_ratio=config["dataset"]["test_split"],
            group_key="scene_id",
            random_seed=config.get("random_seed", 42),
            stratify_key="resolution_tier"
        )
        save_split_manifest(mav_splits, processed_dir / "split_manifest_mavecodd.json")
        for name, items in mav_splits.items():
            groups = sorted({s["scene_id"] for s in items})
            print(f"MaVeCoDD {name:5s}: {len(items):2d} images in {len(groups)} scene groups -> {', '.join(groups)}")

    # 3b. Underwater Crack splits
    crack_samples = scan_crack_samples(raw_crack_dir, zip_fallback=zip_crack)
    if crack_samples:
        crack_splits = create_leakage_free_splits(
            crack_samples,
            train_ratio=config["dataset"]["train_split"],
            val_ratio=config["dataset"]["val_split"],
            test_ratio=config["dataset"]["test_split"],
            group_key="scene_id",
            random_seed=config.get("random_seed", 42)
        )
        save_split_manifest(crack_splits, processed_dir / "split_manifest_crack.json")
        print(f"Underwater Crack Splits -> Train: {len(crack_splits['train'])}, Val: {len(crack_splits['val'])}, Test: {len(crack_splits['test'])}")

    # 3c. Synthetic sample splits
    sample_splits = create_leakage_free_splits(
        sample_items,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_ratio=config["dataset"]["test_split"],
        group_key="scene_id",
        random_seed=config.get("random_seed", 42)
    )
    save_split_manifest(sample_splits, processed_dir / "split_manifest_sample.json")
    print(f"Synthetic Sample Splits -> Train: {len(sample_splits['train'])}, Val: {len(sample_splits['val'])}, Test: {len(sample_splits['test'])}")

    print("=" * 70)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
