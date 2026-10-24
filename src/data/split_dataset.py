"""Dataset split management with strict leakage prevention."""

from typing import Dict, List, Any, Tuple
import os
import json
import numpy as np
from pathlib import Path


def create_leakage_free_splits(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_key: str = "scene_id",
    random_seed: int = 42,
    stratify_key: str | None = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Split samples into train/val/test grouped by scene/sequence to prevent data leakage.

    Args:
        samples: List of sample dictionaries containing metadata and group_key.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        group_key: Key in sample dict indicating inspection scene/sequence group.
        random_seed: Random seed for group shuffling.
        stratify_key: Optional key (e.g. 'resolution_tier'); groups are split separately
            within each stratum so every partition contains every stratum.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to sample lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    rng = np.random.default_rng(random_seed)

    # Group samples by group_key (and stratum, if requested)
    strata: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for s in samples:
        stratum = str(s.get(stratify_key, "all")) if stratify_key else "all"
        g = s.get(group_key, s.get("image_id", "default"))
        strata.setdefault(stratum, {}).setdefault(g, []).append(s)

    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }

    for stratum in sorted(strata):
        grouped = strata[stratum]
        groups = sorted(grouped.keys())
        rng.shuffle(groups)

        n_groups = len(groups)
        if n_groups >= 3:
            # Every partition gets at least one group
            n_test = max(1, int(round(n_groups * test_ratio)))
            n_val = max(1, int(round(n_groups * val_ratio)))
            n_train = max(1, n_groups - n_val - n_test)
            n_val = n_groups - n_train - n_test
        else:
            n_train, n_val = n_groups, 0

        partition = {}
        for i, g in enumerate(groups):
            partition[g] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

        for g in groups:
            splits[partition[g]].extend(grouped[g])

    return splits


def save_split_manifest(splits: Dict[str, List[Dict[str, Any]]], output_path: str | Path) -> None:
    """Save split information to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        split_name: {
            "count": len(items),
            "samples": items
        }
        for split_name, items in splits.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_split_manifest(manifest_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load split manifest from JSON."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {k: v["samples"] for k, v in data.items()}
