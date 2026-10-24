"""Unit tests for dataset loading, mask parsing, and split hygiene."""

import pytest
import numpy as np
from pathlib import Path

from src.data.annotation_parser import parse_mavecodd_mask, parse_crack_mask, extract_defect_boxes
from src.data.preprocessing import resize_image_and_mask, apply_underwater_degradation
from src.data.split_dataset import create_leakage_free_splits
from src.data.dataset_loader import sequence_scene_id
from src.data.preprocessing import extract_patch_grid, label_patch


def test_parse_mavecodd_mask():
    """Test parsing of MaVeCoDD colored corrosion masks into binary format."""
    dummy_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    # Simulate corrosion patch (> 15)
    dummy_mask[20:50, 30:70, :] = [180, 120, 40]
    # Simulate JPEG compression noise (<= 15)
    dummy_mask[0:10, 0:10, :] = [10, 12, 8]

    bin_mask, boxes = parse_mavecodd_mask(dummy_mask, threshold=15)

    assert bin_mask.shape == (100, 100)
    assert bin_mask.dtype == np.uint8
    assert np.all(bin_mask[0:10, 0:10] == 0), "Compression noise should be filtered out"
    assert np.all(bin_mask[25:45, 35:65] == 1), "Defect region should be 1"
    assert len(boxes) >= 1
    assert boxes[0]["defect_type"] == "corrosion"


def test_parse_crack_mask():
    """Test parsing of underwater crack masks."""
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 255

    bin_mask, boxes = parse_crack_mask(dummy_mask, threshold=128)

    assert bin_mask.shape == (100, 100)
    assert bin_mask[50, 50] == 1
    assert bin_mask[10, 10] == 0
    assert len(boxes) == 1
    assert boxes[0]["defect_type"] == "crack"


def test_extract_defect_boxes():
    """Test extraction of normalized bounding box coordinates."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 50:150] = 1

    boxes = extract_defect_boxes(mask, defect_type="corrosion", min_area=50)

    assert len(boxes) == 1
    b = boxes[0]
    assert b["bbox_pixel"] == [50, 50, 150, 100]
    assert b["bbox_normalized"] == [0.25, 0.25, 0.75, 0.5]


def test_underwater_degradation():
    """Test underwater optical degradation preserves array shape and range."""
    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    deg = apply_underwater_degradation(img, turbidity=0.4, red_attenuation=0.7)

    assert deg.shape == (128, 128, 3)
    assert deg.dtype == np.uint8
    # Red channel should be attenuated relative to green/blue
    assert np.mean(deg[:, :, 0]) <= np.mean(deg[:, :, 2])


def test_leakage_free_splits():
    """Test that grouped splitting strictly prevents sequence/scene leakage."""
    samples = [
        {"image_id": f"img_{i}", "scene_id": f"scene_{i // 4}"}
        for i in range(24)
    ]
    splits = create_leakage_free_splits(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_key="scene_id", random_seed=42
    )

    train_scenes = {s["scene_id"] for s in splits["train"]}
    val_scenes = {s["scene_id"] for s in splits["val"]}
    test_scenes = {s["scene_id"] for s in splits["test"]}

    assert len(train_scenes.intersection(val_scenes)) == 0, "Leakage detected between train and val!"
    assert len(train_scenes.intersection(test_scenes)) == 0, "Leakage detected between train and test!"
    assert len(val_scenes.intersection(test_scenes)) == 0, "Leakage detected between val and test!"
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 24


def test_sequence_scene_id_groups_consecutive_frames():
    """Near-duplicate consecutive frames must share a scene group."""
    assert sequence_scene_id("colorImage1005", "LoRes") == sequence_scene_id("colorImage1009", "LoRes")
    assert sequence_scene_id("colorImage1009", "LoRes") != sequence_scene_id("colorImage1010", "LoRes")
    assert sequence_scene_id("hull005", "HiRes") != sequence_scene_id("hull005", "LoRes")


def test_stratified_splits_cover_every_stratum():
    samples = [
        {"image_id": f"{tier}_{i}", "scene_id": f"{tier}_g{i // 3}", "resolution_tier": tier}
        for tier in ("HiRes", "LoRes") for i in range(30)
    ]
    splits = create_leakage_free_splits(samples, 0.6, 0.2, 0.2, group_key="scene_id",
                                        random_seed=0, stratify_key="resolution_tier")
    for name in ("train", "val", "test"):
        assert {s["resolution_tier"] for s in splits[name]} == {"HiRes", "LoRes"}, name
    groups = [{s["scene_id"] for s in splits[n]} for n in ("train", "val", "test")]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert sum(len(v) for v in splits.values()) == 60


def test_patch_grid_labels():
    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    mask = np.zeros((512, 1024), dtype=np.uint8)
    mask[0:256, 0:256] = 1          # first tile fully corroded
    mask[0:5, 256:512] = 1          # second tile: ~2% -> ambiguous
    patches = extract_patch_grid(img, mask, long_side=1024, patch_size=256)
    assert len(patches) == 8
    by_pos = {(p["x"], p["y"]): p["label"] for p in patches}
    assert by_pos[(0, 0)] == 1
    assert by_pos[(256, 0)] == -1
    assert by_pos[(512, 256)] == 0
    assert label_patch(0.05) == 1 and label_patch(0.005) == 0 and label_patch(0.02) == -1
