"""Evaluation runner: both models on train/val/test splits, overfitting check, stress tests."""

import sys
import argparse
import json
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device
from src.data.split_dataset import load_split_manifest
from src.models.baseline import HullDefectBaselineClassifier
from src.models.predict import HullInspectionPredictor
from src.models.evaluate import (
    evaluate_baseline_on_patches,
    evaluate_unet,
    evaluate_underwater_degradation_stress_test,
    overfitting_verdict,
    save_evaluation_plots,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate underwater hull inspection models.")
    parser.add_argument("--dataset", choices=["mavecodd", "crack", "sample"], default="mavecodd",
                        help="Dataset to evaluate on (default: mavecodd)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EVALUATION PIPELINE: AI-POWERED UNDERWATER HULL INSPECTION DRONE [{args.dataset.upper()}]")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("project", {}).get("random_seed", 42))
    dev = get_device(args.device)
    ds = config["dataset"]
    patch_cfg = {
        "long_side": ds.get("patch_long_side", 1024),
        "patch_size": ds.get("patch_size", 256),
        "defect_min_frac": ds.get("patch_defect_min_frac", 0.05),
        "sound_max_frac": ds.get("patch_sound_max_frac", 0.005),
    }

    checkpoints_dir = ROOT / config["paths"]["model_dir"]
    baseline_path = checkpoints_dir / f"baseline_{args.dataset}.joblib"
    unet_path = checkpoints_dir / f"unet_{args.dataset}.pth"
    if not baseline_path.exists() or not unet_path.exists():
        print("Required model checkpoints not found. Run scripts/train_model.py first.")
        sys.exit(1)

    manifest_file = ROOT / config["paths"]["processed_dir"] / f"split_manifest_{args.dataset}.json"
    splits = load_split_manifest(manifest_file)
    print("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    baseline_model = HullDefectBaselineClassifier.load(baseline_path)
    predictor = HullInspectionPredictor(baseline_path=baseline_path, unet_path=unet_path, device=dev)
    print(f"U-Net thresholds (calibrated on val): pixel={predictor.mask_threshold}, "
          f"area={predictor.detection_area_threshold}")

    baseline_eval, unet_eval = {}, {}
    for split in ["train", "val", "test"]:
        print(f"\n[{split.upper()}] Model A (baseline) on patches...")
        baseline_eval[split] = evaluate_baseline_on_patches(baseline_model, splits[split], patch_cfg)
        b = baseline_eval[split]
        print(f"  balanced acc {b['balanced_accuracy']}  acc {b['accuracy']}  P {b['precision']}  "
              f"R {b['recall']}  F1 {b['f1_score']}  ({b['n_patches']} patches: "
              f"{b['n_defect_patches']} defect / {b['n_sound_patches']} sound)")
        print(f"[{split.upper()}] Model B (U-Net)...")
        unet_eval[split] = evaluate_unet(predictor, splits[split], patch_cfg)
        u = unet_eval[split]
        pc = u["patch_classification"]
        print(f"  Dice {u['mean_dice']}  IoU {u['mean_iou']}  | patch balanced acc {pc['balanced_accuracy']}  "
              f"P {pc['precision']}  R {pc['recall']}  F1 {pc['f1_score']}")

    ci = {
        "baseline_patch_balanced_accuracy": baseline_eval["test"].get("balanced_accuracy_ci95"),
        "unet_patch_balanced_accuracy": unet_eval["test"]["patch_classification"].get("balanced_accuracy_ci95"),
        "unet_dice": unet_eval["test"].get("mean_dice_ci95"),
    }
    overfit = {
        "baseline_patch_balanced_accuracy": overfitting_verdict(
            *[baseline_eval[s]["balanced_accuracy"] for s in ("train", "val", "test")]),
        "unet_patch_balanced_accuracy": overfitting_verdict(
            *[unet_eval[s]["patch_classification"]["balanced_accuracy"] for s in ("train", "val", "test")]),
        "unet_dice": overfitting_verdict(*[unet_eval[s]["mean_dice"] for s in ("train", "val", "test")]),
    }
    for k, v in overfit.items():
        v["test_ci95"] = ci[k]
        v["train_within_test_ci"] = bool(ci[k] and ci[k][0] is not None and ci[k][0] <= v["train"] <= ci[k][1])
    print("\nOVERFITTING CHECK (train vs held-out):")
    for k, v in overfit.items():
        print(f"  {k:36s} train {v['train']:.3f} | val {v['val']:.3f} | test {v['test']:.3f} | "
              f"gap {v['gap']:+.3f} -> {v['verdict']} | test 95% CI {v['test_ci95']}")

    print("\nOptical turbidity stress test on TEST split...")
    degradation_results = evaluate_underwater_degradation_stress_test(predictor, splits["test"], patch_cfg)
    for d in degradation_results:
        print(f"  turbidity {d['turbidity']:.2f} ({d['condition']}): U-Net Dice {d['unet_mean_dice']:.3f}, "
              f"U-Net patch bal acc {d['unet_patch_balanced_accuracy']:.3f}, "
              f"baseline patch bal acc {d['baseline_patch_balanced_accuracy']}")

    cross_domain_metrics = None
    crack_manifest = ROOT / config["paths"]["processed_dir"] / "split_manifest_crack.json"
    if crack_manifest.exists() and args.dataset != "crack":
        print("\nCross-domain test: corrosion U-Net on Underwater Crack test images (zero-shot)...")
        crack_test = load_split_manifest(crack_manifest)["test"][:40]
        cross = evaluate_unet(predictor, crack_test, patch_cfg)
        cross_domain_metrics = {
            "source_dataset": args.dataset,
            "target_dataset": "underwater_crack",
            "n_images": len(crack_test),
            "cross_domain_iou": cross["mean_iou"],
            "cross_domain_dice": cross["mean_dice"],
            "cross_domain_pixel_recall": cross["pixel_recall"],
        }
        print(f"  Dice {cross['mean_dice']}  IoU {cross['mean_iou']}  pixel recall {cross['pixel_recall']}")

    summary_file = ROOT / config["paths"]["metrics_dir"] / f"training_summary_{args.dataset}.json"
    history = None
    if summary_file.exists():
        history = json.loads(summary_file.read_text()).get("unet", {}).get("history")

    final_metrics = {
        "dataset": args.dataset,
        "evaluation_protocol": {
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "split_scene_groups": {k: sorted({s["scene_id"] for s in v}) for k, v in splits.items()},
            "patch_config": patch_cfg,
            "unet_mask_threshold": predictor.mask_threshold,
            "unet_detection_area_threshold": predictor.detection_area_threshold,
            "thresholds_chosen_on": "validation split",
        },
        "baseline_model_a": baseline_eval,
        "deep_learning_model_b": unet_eval,
        "overfitting_check": overfit,
        "optical_degradation_stress_test": degradation_results,
        "cross_domain_generalization": cross_domain_metrics,
    }

    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    figures_dir = ROOT / config["paths"]["figures_dir"]
    if args.dataset != "mavecodd":
        # keep the primary (MaVeCoDD) figures from being overwritten by quick-start runs
        figures_dir = figures_dir / args.dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = metrics_dir / f"test_metrics_{args.dataset}.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    save_evaluation_plots(final_metrics, history, figures_dir)

    print("\n" + "=" * 70)
    print("EVALUATION FINISHED!")
    print(f"Metrics saved to: {metrics_out}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
