"""Model evaluation: Model A vs Model B on the same patches, segmentation quality,
overfitting (train vs val vs test) analysis, turbidity stress test and cross-domain test.

Why patch-level classification: every MaVeCoDD image contains some corrosion, so an
image-level "defect / no defect" score is meaningless (always answering "defect" scores
100%). Instead both models are scored on the same grid of 256 px tiles, where tiles are
labelled defect (>= 5% corrosion) or sound (<= 0.5%) from the ground-truth mask.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.data.dataset_loader import load_image_and_mask
from src.data.preprocessing import (
    apply_underwater_degradation,
    resize_long_side,
    tile_positions,
    label_patch,
)
from src.models.baseline import HullDefectBaselineClassifier, prepare_patch_dataset
from src.models.cnn import compute_segmentation_metrics
from src.models.predict import HullInspectionPredictor


def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    """Standard binary classification metrics with an explicit confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    auc = None
    if y_score is not None and len(np.unique(y_true)) > 1:
        auc = round(float(roc_auc_score(y_true, y_score)), 4)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4) if len(np.unique(y_true)) > 1 else None,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": auc,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "false_negatives": fn,
        "false_positives": fp,
        "n_patches": int(len(y_true)),
        "n_defect_patches": int(np.sum(y_true == 1)),
        "n_sound_patches": int(np.sum(y_true == 0)),
    }


def bootstrap_ci(per_group_fn, groups, n_boot: int = 1000, seed: int = 0) -> List[float]:
    """95% cluster-bootstrap interval: resample whole IMAGES with replacement (tiles from one
    image are correlated, so resampling tiles would understate the uncertainty)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        v = per_group_fn([uniq[i] for i in pick])
        if v is not None:
            vals.append(v)
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def patch_balanced_accuracy_ci(y_true, y_pred, image_ids) -> List[float]:
    y_true, y_pred, image_ids = np.asarray(y_true), np.asarray(y_pred), np.asarray(image_ids)
    idx_by_img = {g: np.where(image_ids == g)[0] for g in set(image_ids.tolist())}

    def stat(sample):
        idx = np.concatenate([idx_by_img[g] for g in sample])
        yt = y_true[idx]
        return balanced_accuracy_score(yt, y_pred[idx]) if len(np.unique(yt)) > 1 else None

    return bootstrap_ci(stat, image_ids.tolist())


def unet_patch_scores(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    mask_threshold: float,
    long_side: int = 1024,
    patch_size: int = 256,
    defect_min_frac: float = 0.05,
    sound_max_frac: float = 0.005
) -> Tuple[List[int], List[float]]:
    """Ground-truth tile labels and the U-Net's predicted defect-area fraction per tile.

    Uses exactly the same tile grid and labelling rule as the baseline's patch dataset,
    so both models are scored on identical tiles.
    """
    h0, w0 = gt_mask.shape[:2]
    scale = long_side / float(max(h0, w0))
    w, h = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    pred = cv2.resize((prob_map >= mask_threshold).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    gt = (cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    if min(h, w) < patch_size:
        ph, pw = max(0, patch_size - h), max(0, patch_size - w)
        pred = cv2.copyMakeBorder(pred, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        gt = cv2.copyMakeBorder(gt, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        h, w = gt.shape[:2]

    y_true, frac = [], []
    for y in tile_positions(h, patch_size):
        for x in tile_positions(w, patch_size):
            lbl = label_patch(float(gt[y:y + patch_size, x:x + patch_size].mean()), defect_min_frac, sound_max_frac)
            if lbl < 0:
                continue
            y_true.append(lbl)
            frac.append(float(pred[y:y + patch_size, x:x + patch_size].mean()))
    return y_true, frac


def evaluate_baseline_on_patches(
    baseline_model: HullDefectBaselineClassifier,
    samples: List[Dict[str, Any]],
    patch_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate Model A on the shared patch grid of the given scenes."""
    imgs, y_true, meta = prepare_patch_dataset(samples, return_meta=True, **patch_cfg)
    if not imgs:
        return {}
    t0 = time.perf_counter()
    X = baseline_model.extract_batch_features(imgs, [m["context"] for m in meta])
    y_pred = baseline_model.predict(X)
    y_prob = baseline_model.predict_proba(X)[:, 1]
    latency = (time.perf_counter() - t0) * 1000.0 / len(imgs)
    out = classification_metrics(y_true, y_pred, y_prob)
    out["balanced_accuracy_ci95"] = patch_balanced_accuracy_ci(y_true, y_pred, [m["image_id"] for m in meta])
    out["model_name"] = "Model A (colour/texture features + SVM)"
    out["latency_ms_per_patch"] = round(latency, 2)
    return out


def evaluate_unet(
    predictor: HullInspectionPredictor,
    samples: List[Dict[str, Any]],
    patch_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate Model B: per-image segmentation quality + patch classification on shared tiles."""
    per_image = []
    y_true_all, frac_all, ids_all = [], [], []
    latencies = []
    for s in samples:
        img, gt = load_image_and_mask(s)
        res = predictor.predict(img, run_baseline=False)
        latencies.append(res["inference_time_ms"])
        m = compute_segmentation_metrics(res["probability_map"], gt, threshold=predictor.mask_threshold)
        per_image.append({
            "image_id": s.get("image_id"),
            "scene_id": s.get("scene_id"),
            "true_defect_area": round(float(gt.mean()), 4),
            "pred_defect_area": round(float(res["defect_area_fraction"]), 4),
            **{k: round(v, 4) for k, v in m.items()},
        })
        yt, fr = unet_patch_scores(res["probability_map"], gt, predictor.mask_threshold, **patch_cfg)
        y_true_all += yt
        frac_all += fr
        ids_all += [s.get("image_id")] * len(yt)

    frac_arr = np.array(frac_all)
    pred_arr = (frac_arr >= predictor.detection_area_threshold).astype(int)
    patch = classification_metrics(y_true_all, pred_arr, frac_arr)
    patch["balanced_accuracy_ci95"] = patch_balanced_accuracy_ci(y_true_all, pred_arr, ids_all)
    dice_by_img = {p["image_id"]: p["dice"] for p in per_image}
    return {
        "model_name": "Model B (U-Net CNN)",
        "n_images": len(samples),
        "mean_iou": round(float(np.mean([p["iou"] for p in per_image])), 4),
        "mean_dice": round(float(np.mean([p["dice"] for p in per_image])), 4),
        "mean_dice_ci95": bootstrap_ci(lambda sample: float(np.mean([dice_by_img[g] for g in sample])),
                                       list(dice_by_img)),
        "mean_pixel_accuracy": round(float(np.mean([p["pixel_accuracy"] for p in per_image])), 4),
        "pixel_precision": round(float(np.mean([p["precision"] for p in per_image])), 4),
        "pixel_recall": round(float(np.mean([p["recall"] for p in per_image])), 4),
        "patch_classification": patch,
        "per_image": per_image,
        "latency_ms_per_image": round(float(np.mean(latencies)), 2),
    }


def overfitting_verdict(train_score: float, val_score: float, test_score: float) -> Dict[str, Any]:
    """Classify the train -> held-out generalisation gap.

    gap < 0.05: no meaningful overfitting; 0.05-0.10: mild; > 0.10: overfitting.
    The held-out score is the mean of validation and test.
    """
    held_out = (val_score + test_score) / 2.0
    gap = train_score - held_out
    if gap < 0.05:
        verdict = "No meaningful overfitting"
    elif gap < 0.10:
        verdict = "Mild overfitting"
    else:
        verdict = "Overfitting"
    return {"train": round(train_score, 4), "val": round(val_score, 4), "test": round(test_score, 4),
            "gap": round(gap, 4), "verdict": verdict}


def evaluate_underwater_degradation_stress_test(
    predictor: HullInspectionPredictor,
    test_samples: List[Dict[str, Any]],
    patch_cfg: Dict[str, Any],
    turbidity_levels: List[float] = [0.0, 0.15, 0.30, 0.45, 0.60]
) -> List[Dict[str, Any]]:
    """Both models on artificially degraded test images (fixed seeds -> reproducible)."""
    names = {0.0: "Clear Water", 0.15: "Light Coastal", 0.30: "Harbour Waters",
             0.45: "Turbid Estuarine", 0.60: "Heavy Turbidity"}
    cached = []
    for s in test_samples:
        img, gt = load_image_and_mask(s)
        cached.append(resize_long_side(img, gt, patch_cfg["long_side"]))

    results = []
    for turb in turbidity_levels:
        dices, recalls = [], []
        b_true, b_pred, u_true, u_frac = [], [], [], []
        for i, (img, gt) in enumerate(cached):
            deg = apply_underwater_degradation(img, turbidity=turb, red_attenuation=0.65, random_seed=1000 + i) if turb > 0 else img
            res = predictor.predict(deg, run_baseline=predictor.baseline_model is not None, gt_mask=gt)
            m = compute_segmentation_metrics(res["probability_map"], gt, threshold=predictor.mask_threshold)
            dices.append(m["dice"])
            recalls.append(m["recall"])
            yt, fr = unet_patch_scores(res["probability_map"], gt, predictor.mask_threshold, **patch_cfg)
            u_true += yt
            u_frac += fr
            if res["baseline_result"]:
                for t in res["baseline_result"]["tiles"]:
                    if t["true_label"] >= 0:
                        b_true.append(t["true_label"])
                        b_pred.append(t["prediction"])
        u_pred = (np.array(u_frac) >= predictor.detection_area_threshold).astype(int)
        results.append({
            "turbidity": turb,
            "condition": names.get(turb, f"Turbidity {turb:.2f}"),
            "unet_mean_dice": round(float(np.mean(dices)), 4),
            "unet_pixel_recall": round(float(np.mean(recalls)), 4),
            "unet_patch_balanced_accuracy": round(float(balanced_accuracy_score(u_true, u_pred)), 4),
            "baseline_patch_balanced_accuracy": round(float(balanced_accuracy_score(b_true, b_pred)), 4) if b_true else None,
        })
    return results


def save_evaluation_plots(metrics: Dict[str, Any], history: Optional[Dict[str, List[float]]], output_dir: str | Path) -> None:
    """Static PNG figures (used in reports); the dashboard draws interactive versions."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

    # 1. Patch-level confusion matrices on the TEST split (same tiles for both models)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title, cmap in [
        (axes[0], metrics["baseline_model_a"]["test"], "Model A: features + SVM", "Blues"),
        (axes[1], metrics["deep_learning_model_b"]["test"]["patch_classification"], "Model B: U-Net", "Greens"),
    ]:
        c = key["confusion_matrix"]
        cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        ax.imshow(cm, cmap=cmap)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Sound", "Defect"])
        ax.set_yticks([0, 1], ["Sound", "Defect"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nTest patches (balanced acc {key['balanced_accuracy']:.2f})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_comparison.png", dpi=200)
    plt.close()

    # 2. Train / Val / Test comparison (overfitting check)
    of = metrics["overfitting_check"]
    labels = ["Model A\npatch balanced acc", "Model B\npatch balanced acc", "Model B\nsegmentation Dice"]
    rows = [of["baseline_patch_balanced_accuracy"], of["unet_patch_balanced_accuracy"], of["unet_dice"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (split, color) in enumerate([("train", "#7f8c8d"), ("val", "#3498db"), ("test", "#27ae60")]):
        vals = [r[split] for r in rows]
        ax.bar(x + (k - 1) * 0.26, vals, 0.26, label=split.title(), color=color)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 1) * 0.26, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Train vs Validation vs Test (small gap = no overfitting)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "model_comparison_barchart.png", dpi=200)
    plt.close()

    # 3. Learning curves
    if history and history.get("epoch"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history["epoch"], history["train_loss"], "o-", label="Train")
        axes[0].plot(history["epoch"], history["val_loss"], "s-", label="Validation")
        axes[0].set_title("U-Net loss per epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["train_dice"], "o-", label="Train")
        axes[1].plot(history["epoch"], history["val_dice"], "s-", label="Validation")
        axes[1].set_title("U-Net Dice per epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out / "unet_learning_curves.png", dpi=200)
        plt.close()

    # 4. Turbidity curve
    deg = metrics.get("optical_degradation_stress_test") or []
    if deg:
        t = [d["turbidity"] for d in deg]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(t, [d["unet_mean_dice"] for d in deg], "o-", label="U-Net Dice")
        ax.plot(t, [d["unet_patch_balanced_accuracy"] for d in deg], "s-", label="U-Net patch balanced acc")
        if deg[0].get("baseline_patch_balanced_accuracy") is not None:
            ax.plot(t, [d["baseline_patch_balanced_accuracy"] for d in deg], "^-", label="Baseline patch balanced acc")
        ax.set_xlabel("Simulated turbidity")
        ax.set_ylim(0, 1.05)
        ax.set_title("Robustness to underwater degradation (test split)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "turbidity_degradation_curve.png", dpi=200)
        plt.close()
