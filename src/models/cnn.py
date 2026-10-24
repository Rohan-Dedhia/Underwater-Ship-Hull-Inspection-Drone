"""Deep Learning Architecture: Lightweight Multi-Task U-Net for Hull Defect Segmentation and Detection."""

from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessing import resize_image_and_mask
from src.data.annotation_parser import extract_defect_boxes


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 with residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class HullUNet(nn.Module):
    """Lightweight U-Net architecture with dual segmentation and classification heads."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 8, base_channels * 16),
            nn.Dropout2d(0.2)
        )

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_channels * 2, base_channels)

        # Segmentation mask head
        self.mask_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Image-level classification head from bottleneck
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(base_channels * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            mask_logits: [B, 1, H, W] raw segmentation logits.
            class_logits: [B, 1] raw classification logits.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck(x4)

        # Classification branch
        pooled = self.global_pool(b).flatten(1)
        class_logits = self.classifier_head(pooled)

        # Segmentation branch
        u1 = self.up1(b)
        u1 = self.conv_up1(torch.cat([u1, x4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, x3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, x2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv_up4(torch.cat([u4, x1], dim=1))

        mask_logits = self.mask_head(u4)
        return mask_logits, class_logits


class CombinedSegmentationLoss(nn.Module):
    """Combined BCE + Soft Dice + Auxiliary Classification Loss for imbalanced defect segmentation."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mask_logits: torch.Tensor,
        target_mask: torch.Tensor,
        class_logits: torch.Tensor,
        target_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Mask BCE Loss
        mask_bce = self.bce(mask_logits, target_mask)

        # 2. Soft Dice Loss
        mask_probs = torch.sigmoid(mask_logits)
        intersection = (mask_probs * target_mask).sum(dim=(2, 3))
        union = mask_probs.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # 3. Auxiliary Classification BCE Loss
        class_bce = self.bce(class_logits, target_label)

        total_loss = self.alpha * mask_bce + self.beta * dice_loss + self.gamma * class_bce

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "mask_bce": float(mask_bce.item()),
            "dice_loss": float(dice_loss.item()),
            "class_bce": float(class_bce.item())
        }
        return total_loss, loss_dict


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> Dict[str, float]:
    """Calculate IoU, Dice (F1), Pixel Accuracy, Precision, and Recall."""
    bin_pred = (pred_mask > threshold).astype(np.uint8)
    bin_target = (target_mask > 0.5).astype(np.uint8)

    intersection = np.sum(bin_pred * bin_target)
    union = np.sum(np.clip(bin_pred + bin_target, 0, 1))
    pred_sum = np.sum(bin_pred)
    target_sum = np.sum(bin_target)

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    pixel_acc = np.mean(bin_pred == bin_target)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall)
    }


def predict_probability_map(
    model: HullUNet,
    image: np.ndarray,
    work_long_side: int = 512,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Run the U-Net on a full RGB image and return a per-pixel defect probability map.

    The image is resized (aspect ratio kept) so its long side is `work_long_side`,
    padded to a multiple of 16, passed through the network, then the padding is removed
    and the map is resized back to the ORIGINAL image resolution.
    """
    from src.data.preprocessing import resize_long_side, pad_to_multiple, to_normalized_tensor_array

    dev = device or next(model.parameters()).device
    h0, w0 = image.shape[:2]
    work, _ = resize_long_side(image, None, work_long_side)
    padded, (h, w) = pad_to_multiple(work, 16)
    x = torch.from_numpy(to_normalized_tensor_array(padded)).unsqueeze(0).to(dev)
    model.eval()
    with torch.no_grad():
        mask_logits, _ = model(x)
    prob = torch.sigmoid(mask_logits)[0, 0, :h, :w].cpu().numpy()
    return cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
