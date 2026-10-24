"""Training routines for both Model A (Baseline SVM) and Model B (U-Net).

Overfitting safeguards used here:
- Scene-grouped splits (near-duplicate frames never cross train/val/test).
- Model A: hyperparameters chosen by scene-grouped K-fold CV on the training split with the
  one-standard-error rule (prefers the simpler / more regularised model).
- Model B: small network (~2M params), random-crop + zoom + flip + colour + turbidity
  augmentation, weight decay, dropout, and early stopping on validation loss.
- Train AND validation metrics are recorded every epoch so the train/val gap is visible.
- Decision thresholds are calibrated on the validation split only (never on test).
"""

from typing import Dict, Any, List, Optional
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.utils.reproducibility import get_device, set_seed
from src.data.dataset_loader import HullDefectDataset, load_image_and_mask
from src.models.baseline import (
    HullDefectBaselineClassifier,
    prepare_patch_dataset,
    select_hyperparameters_by_group_cv,
)
from src.models.cnn import (
    HullUNet,
    CombinedSegmentationLoss,
    compute_segmentation_metrics,
    predict_probability_map,
)


def _patch_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ds = cfg.get("dataset", {})
    return {
        "long_side": ds.get("patch_long_side", 1024),
        "patch_size": ds.get("patch_size", 256),
        "defect_min_frac": ds.get("patch_defect_min_frac", 0.05),
        "sound_max_frac": ds.get("patch_sound_max_frac", 0.005),
    }


def train_baseline(
    train_samples: List[Dict[str, Any]],
    val_samples: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Train the traditional ML baseline (colour/texture features + SVM) on scene patches."""
    t0 = time.time()
    cfg = config or {}
    base_cfg = cfg.get("baseline_model", {})
    pcfg = _patch_cfg(cfg)
    seed = cfg.get("project", {}).get("random_seed", 42)

    print("\n--- Training Traditional ML Baseline (Model A) ---")
    print(f"Tiling {len(train_samples)} training scenes into {pcfg['patch_size']}px patches...")
    train_imgs, train_y, train_meta = prepare_patch_dataset(train_samples, return_meta=True, **pcfg)
    print(f"Training patches: {len(train_imgs)} (Defect: {int(np.sum(train_y == 1))}, Sound: {int(np.sum(train_y == 0))})")

    extractor = HullDefectBaselineClassifier()
    X_train = extractor.extract_batch_features(train_imgs, [m["context"] for m in train_meta])
    print(f"Feature matrix shape: {X_train.shape}")

    groups = [m["scene_id"] for m in train_meta]
    n_comp_cap = min(X_train.shape[0] - 1, X_train.shape[1])
    pca_grid = [c for c in base_cfg.get("cv_pca_components", [None]) if c is None or c < n_comp_cap] or [None]
    print(f"Scene-grouped CV over C={base_cfg.get('cv_svm_c')} x PCA={pca_grid} ({len(set(groups))} scene groups)...")
    cv_res = select_hyperparameters_by_group_cv(
        X_train, train_y, groups,
        c_values=base_cfg.get("cv_svm_c", [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
        pca_values=pca_grid,
        n_folds=base_cfg.get("cv_folds", 5),
        random_state=seed,
    )
    chosen = cv_res["chosen"]
    print(f"Best CV:   C={cv_res['best']['C']}, PCA={cv_res['best']['pca_components']} -> "
          f"balanced acc {cv_res['best']['cv_mean']:.3f} +/- {cv_res['best']['cv_std']:.3f}")
    print(f"Chosen (1-SE rule): C={chosen['C']}, PCA={chosen['pca_components']} -> "
          f"CV {chosen['cv_mean']:.3f}, train {chosen['train_mean']:.3f}")

    model = HullDefectBaselineClassifier(
        use_pca=chosen["pca_components"] is not None,
        pca_components=chosen["pca_components"] or 16,
        svm_c=chosen["C"],
        svm_gamma="scale",
        svm_kernel="rbf",
        random_state=seed,
    )
    model.fit(X_train, train_y)
    model.patch_config = pcfg

    train_pred = model.predict(X_train)
    train_bal_acc = float(balanced_accuracy_score(train_y, train_pred))
    print(f"Train balanced accuracy: {train_bal_acc:.4f}")

    val_bal_acc = None
    if val_samples:
        val_imgs, val_y, val_meta = prepare_patch_dataset(val_samples, return_meta=True, **pcfg)
        if val_imgs:
            val_pred = model.predict(model.extract_batch_features(val_imgs, [m["context"] for m in val_meta]))
            val_bal_acc = float(balanced_accuracy_score(val_y, val_pred))
            print(f"Validation balanced accuracy: {val_bal_acc:.4f} ({len(val_imgs)} patches)")

    training_time = time.time() - t0
    print(f"Baseline training completed in {training_time:.2f} seconds.")

    if save_path:
        model.save(save_path)
        print(f"Saved baseline checkpoint to {save_path}")

    return {
        "model": model,
        "training_time_s": training_time,
        "n_train_patches": len(train_imgs),
        "train_balanced_accuracy": train_bal_acc,
        "val_balanced_accuracy": val_bal_acc,
        "val_accuracy": val_bal_acc,
        "feature_dim": int(X_train.shape[1]),
        "cv": cv_res,
    }


def _evaluate_loader(model, loader, criterion, dev) -> Dict[str, float]:
    """Loss + mean IoU/Dice over full (un-augmented) images, at the 512px working resolution."""
    model.eval()
    losses, ious, dices = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, labels = batch["image"].to(dev), batch["mask"].to(dev), batch["label"].to(dev)
            h, w = [int(v) for v in batch["valid_hw"][0]]
            mask_logits, class_logits = model(imgs)
            mask_logits, masks = mask_logits[..., :h, :w], masks[..., :h, :w]
            loss, _ = criterion(mask_logits, masks, class_logits, labels)
            losses.append(loss.item())
            m = compute_segmentation_metrics(torch.sigmoid(mask_logits)[0, 0].cpu().numpy(), masks[0, 0].cpu().numpy())
            ious.append(m["iou"])
            dices.append(m["dice"])
    return {"loss": float(np.mean(losses)), "iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def calibrate_unet_thresholds(
    model: HullUNet,
    val_samples: List[Dict[str, Any]],
    work_long_side: int,
    patch_cfg: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Choose the pixel threshold (max mean Dice) and the patch-area threshold (max balanced
    accuracy) on the VALIDATION split. Test data is never used for any choice."""
    from src.models.evaluate import unet_patch_scores

    prob_maps, masks = [], []
    for s in val_samples:
        img, mask = load_image_and_mask(s)
        prob_maps.append(predict_probability_map(model, img, work_long_side, device))
        masks.append(mask)

    pixel_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    dice_by_thr = {}
    for thr in pixel_grid:
        dice_by_thr[thr] = float(np.mean([compute_segmentation_metrics(p, m, threshold=thr)["dice"]
                                         for p, m in zip(prob_maps, masks)]))
    mask_thr = max(dice_by_thr, key=dice_by_thr.get)

    y_true, frac = [], []
    for p, m in zip(prob_maps, masks):
        yt, fr = unet_patch_scores(p, m, mask_thr, **patch_cfg)
        y_true += yt
        frac += fr
    y_true, frac = np.array(y_true), np.array(frac)
    area_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    bal_by_thr = {}
    for a in area_grid:
        bal_by_thr[a] = float(balanced_accuracy_score(y_true, (frac >= a).astype(int))) if len(np.unique(y_true)) > 1 else 0.0
    area_thr = max(bal_by_thr, key=bal_by_thr.get)

    print(f"Calibrated on validation: pixel threshold={mask_thr} (val Dice {dice_by_thr[mask_thr]:.4f}), "
          f"patch area threshold={area_thr} (val patch balanced acc {bal_by_thr[area_thr]:.4f})")
    return {
        "mask_threshold": mask_thr,
        "patch_area_threshold": area_thr,
        "val_dice_by_threshold": dice_by_thr,
        "val_patch_bal_acc_by_area_threshold": bal_by_thr,
    }


def train_unet(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    resume_path: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Train Deep Learning U-Net (Model B) for segmentation and detection.

    After every epoch the full training state (model, optimizer, scheduler, early-stopping
    counters, history, RNG states) is written to `<save_path stem>_last.pth`. Passing that file
    as `resume_path` continues the run from the next epoch. Passing a final checkpoint (weights
    only) instead warm-starts a new run from those weights.
    """
    cfg = config or {}
    cnn_cfg = cfg.get("cnn_model", {})
    epochs = cnn_cfg.get("epochs", 30)
    batch_size = cnn_cfg.get("batch_size", 8)
    lr = cnn_cfg.get("learning_rate", 1e-3)
    patience = cnn_cfg.get("early_stopping_patience", 7)
    base_channels = cnn_cfg.get("base_channels", 16)
    work_long_side = cnn_cfg.get("work_long_side", 512)
    seed = cfg.get("project", {}).get("random_seed", 42)
    set_seed(seed)

    dev = device or get_device(cnn_cfg.get("device", "auto"))
    print(f"\n--- Training Deep Learning U-Net (Model B) on {dev} ---")
    print("Caching resized images...")

    train_ds = HullDefectDataset(
        train_samples, mode="train", work_long_side=work_long_side,
        crop_size=cnn_cfg.get("crop_size", 256),
        crops_per_image=cnn_cfg.get("crops_per_image", 2),
        scale_aug=tuple(cnn_cfg.get("scale_aug", [0.8, 1.6])),
        degrade_prob=0.3, random_seed=seed,
    )
    train_eval_ds = HullDefectDataset(train_samples, mode="eval", work_long_side=work_long_side)
    val_ds = HullDefectDataset(val_samples, mode="eval", work_long_side=work_long_side)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = HullUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M | train crops/epoch: {len(train_ds)} | val images: {len(val_ds)}")
    criterion = CombinedSegmentationLoss(alpha=1.0, beta=1.2, gamma=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cnn_cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {k: [] for k in ["epoch", "train_aug_loss", "train_loss", "train_iou", "train_dice",
                               "val_loss", "val_iou", "val_dice", "lr"]}

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    no_improve = 0
    stopped_early = False
    start_epoch = 1
    elapsed_before = 0.0

    last_path = Path(save_path).with_name(Path(save_path).stem + "_last.pth") if save_path else None

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scheduler.T_max = epochs  # allow extending the run with a larger --epochs
            history = ckpt["history"]
            best_val_loss = ckpt["best_val_loss"]
            best_weights = ckpt["best_weights"]
            best_epoch = ckpt["best_epoch"]
            no_improve = ckpt["no_improve"]
            elapsed_before = ckpt.get("elapsed_s", 0.0)
            start_epoch = ckpt["last_epoch"] + 1
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            g.set_state(ckpt["rng"]["loader"])
            train_ds.rng.bit_generator.state = ckpt["rng"]["dataset"]
            print(f"Resuming from {resume_path} at epoch {start_epoch} (best epoch so far {best_epoch}).")
        else:
            print(f"Warm-starting from weights in {resume_path} (no optimizer state; starting a new schedule).")

    t0 = time.time() - elapsed_before

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        aug_loss_accum = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(dev)
            masks = batch["mask"].to(dev)
            labels = batch["label"].to(dev)

            optimizer.zero_grad()
            mask_logits, class_logits = model(imgs)
            loss, _ = criterion(mask_logits, masks, class_logits, labels)
            loss.backward()
            optimizer.step()
            aug_loss_accum += loss.item() * imgs.size(0)

        curr_lr = float(optimizer.param_groups[0]["lr"])
        tr = _evaluate_loader(model, train_eval_loader, criterion, dev)
        va = _evaluate_loader(model, val_loader, criterion, dev)

        history["epoch"].append(epoch)
        history["train_aug_loss"].append(aug_loss_accum / len(train_ds))
        history["train_loss"].append(tr["loss"])
        history["train_iou"].append(tr["iou"])
        history["train_dice"].append(tr["dice"])
        history["val_loss"].append(va["loss"])
        history["val_iou"].append(va["iou"])
        history["val_dice"].append(va["dice"])
        history["lr"].append(curr_lr)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss {tr['loss']:.4f} Dice {tr['dice']:.4f} | "
              f"Val Loss {va['loss']:.4f} Dice {va['dice']:.4f} | gap {tr['dice'] - va['dice']:+.4f} | "
              f"LR {curr_lr:.2e} | {time.time() - t0:.0f}s", flush=True)

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}: val loss has not improved for {patience} epochs "
                      f"(best epoch {best_epoch}).")
                stopped_early = True
                break

        scheduler.step()

        if last_path:
            last_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "last_epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_weights": best_weights,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "history": history,
                "elapsed_s": time.time() - t0,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "loader": g.get_state(),
                    "dataset": train_ds.rng.bit_generator.state,
                },
                "config": cnn_cfg,
            }, last_path)

    total_time = time.time() - t0
    print(f"U-Net training completed in {total_time:.1f} s. Restoring best epoch {best_epoch}.")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    calib = calibrate_unet_thresholds(model, val_samples, work_long_side, _patch_cfg(cfg), dev)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "base_channels": base_channels,
            "work_long_side": work_long_side,
            "mask_threshold": calib["mask_threshold"],
            "patch_area_threshold": calib["patch_area_threshold"],
            "patch_config": _patch_cfg(cfg),
            "history": history,
            "config": cnn_cfg,
        }, save_path)
        print(f"Saved U-Net checkpoint to {save_path}")

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "n_params": n_params,
        "calibration": calib,
        "training_time_s": total_time,
        "device": str(dev),
    }
