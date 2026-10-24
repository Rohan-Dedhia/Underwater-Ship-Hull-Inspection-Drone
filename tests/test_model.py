"""Unit tests for ML models: baseline feature extraction, U-Net architecture, and losses."""

import pytest
import numpy as np
import torch

from src.models.baseline import HullDefectBaselineClassifier
from src.models.cnn import HullUNet, CombinedSegmentationLoss, compute_segmentation_metrics, predict_probability_map


def test_baseline_feature_extraction():
    """Handcrafted colour/texture feature vector: fixed length, finite, context-aware."""
    from src.data.preprocessing import image_color_context

    classifier = HullDefectBaselineClassifier()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    feat = classifier.extract_features(img)

    assert feat.ndim == 1
    assert len(feat) == 47
    assert np.isfinite(feat).all()

    # The same patch inside a differently coloured image must give different relative-colour features
    bright = np.full((512, 512, 3), 230, dtype=np.uint8)
    feat_ctx = classifier.extract_features(img, image_color_context(bright))
    assert not np.allclose(feat[:15], feat_ctx[:15])


def test_baseline_learns_rust_vs_paint():
    """End-to-end sanity check: rust-coloured rough patches vs smooth paint are separable."""
    rng = np.random.default_rng(1)
    paint = [np.clip(rng.normal([120, 40, 35], 6, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(20)]
    rust = [np.clip(rng.normal([170, 90, 30], 35, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(20)]
    clf = HullDefectBaselineClassifier(svm_c=1.0)
    X = clf.extract_batch_features(paint + rust)
    y = np.array([0] * 20 + [1] * 20)
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.9
    proba = clf.predict_proba(X)
    assert proba.shape == (40, 2) and np.all((proba >= 0) & (proba <= 1))


def test_unet_forward_pass():
    """Test U-Net dual-head output tensor shapes."""
    model = HullUNet(in_channels=3, out_channels=1, base_channels=16)
    model.eval()

    # Batch of 2 RGB images at 256x256
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        mask_logits, class_logits = model(x)

    # Segmentation head: [B, 1, H, W]
    assert mask_logits.shape == (2, 1, 256, 256)
    # Classification head: [B, 1]
    assert class_logits.shape == (2, 1)


def test_combined_loss_backward():
    """Test that combined BCE + Dice loss produces valid gradients."""
    model = HullUNet(in_channels=3, out_channels=1, base_channels=16)
    criterion = CombinedSegmentationLoss()

    x = torch.randn(2, 3, 128, 128)
    target_mask = torch.randint(0, 2, (2, 1, 128, 128)).float()
    target_label = torch.tensor([[1.0], [0.0]])

    mask_logits, class_logits = model(x)
    loss, loss_dict = criterion(mask_logits, target_mask, class_logits, target_label)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    loss.backward()

    # Check that parameters received gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads


def test_segmentation_metrics():
    """Test IoU and Dice metric calculation against known values."""
    pred = np.zeros((10, 10), dtype=np.float32)
    target = np.zeros((10, 10), dtype=np.float32)

    # 4 overlapping pixels out of 6 total
    pred[2:4, 2:4] = 1.0   # 4 pixels
    target[2:4, 2:4] = 1.0 # 4 pixels
    target[2:4, 4:5] = 1.0 # +2 pixels -> 6 total

    metrics = compute_segmentation_metrics(pred, target, threshold=0.5)

    assert 0.6 < metrics["iou"] < 0.7   # 4 / 6 = 0.667
    assert 0.7 < metrics["dice"] < 0.85 # (2*4) / (4+6) = 0.80
    assert metrics["precision"] > 0.95


def test_predict_probability_map_keeps_original_resolution():
    """U-Net inference on a non-square image returns a map at the ORIGINAL resolution."""
    model = HullUNet(base_channels=8)
    img = np.random.randint(0, 256, (300, 700, 3), dtype=np.uint8)
    prob = predict_probability_map(model, img, work_long_side=256, device=torch.device("cpu"))
    assert prob.shape == (300, 700)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_overfitting_verdict_thresholds():
    from src.models.evaluate import overfitting_verdict
    assert overfitting_verdict(0.80, 0.79, 0.78)["verdict"] == "No meaningful overfitting"
    assert overfitting_verdict(0.90, 0.83, 0.83)["verdict"] == "Mild overfitting"
    assert overfitting_verdict(1.00, 0.70, 0.70)["verdict"] == "Overfitting"
