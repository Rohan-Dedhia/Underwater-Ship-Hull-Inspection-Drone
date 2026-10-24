"""Unified inference predictor for hull defect detection and segmentation."""

from typing import Dict, Any, Optional, Union, Tuple
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from src.utils.reproducibility import get_device
from src.data.annotation_parser import extract_defect_boxes
from src.models.baseline import HullDefectBaselineClassifier
from src.models.cnn import HullUNet, predict_probability_map


class HullInspectionPredictor:
    """Inference wrapper serving both Model A (patch classifier) and Model B (U-Net segmenter).

    Decision rule (Model B): a pixel is 'defect' if its probability >= mask_threshold, and the
    image/frame is flagged if the predicted defect area >= detection_area_threshold of the frame.
    Both thresholds are calibrated on the validation split during training and stored in the
    checkpoint; they can be overridden explicitly.
    """

    def __init__(
        self,
        baseline_path: Optional[str | Path] = None,
        unet_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        mask_threshold: Optional[float] = None,
        detection_area_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.device = device or get_device("auto")
        self.mask_threshold = 0.5
        self.detection_area_threshold = 0.02
        self.work_long_side = 512
        self.patch_config = {"long_side": 1024, "patch_size": 256}
        self.unet_info: Dict[str, Any] = {}

        # Load Model A (Baseline)
        self.baseline_model = None
        if baseline_path and Path(baseline_path).exists():
            try:
                self.baseline_model = HullDefectBaselineClassifier.load(baseline_path)
                pc = getattr(self.baseline_model, "patch_config", None)
                if pc:
                    self.patch_config = {"long_side": pc["long_side"], "patch_size": pc["patch_size"]}
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")

        # Load Model B (U-Net)
        self.unet_model = None
        if unet_path and Path(unet_path).exists():
            try:
                ckpt = torch.load(unet_path, map_location=self.device, weights_only=False)
                state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                meta = ckpt if isinstance(ckpt, dict) else {}
                base_channels = meta.get("base_channels", state["inc.conv.0.weight"].shape[0])
                self.unet_model = HullUNet(in_channels=3, out_channels=1, base_channels=base_channels)
                self.unet_model.load_state_dict(state)
                self.unet_model.to(self.device)
                self.unet_model.eval()
                self.work_long_side = meta.get("work_long_side", 256)
                self.mask_threshold = meta.get("mask_threshold", self.mask_threshold)
                self.detection_area_threshold = meta.get("patch_area_threshold", self.detection_area_threshold)
                self.unet_info = {k: meta.get(k) for k in ("epoch", "base_channels", "work_long_side",
                                                           "mask_threshold", "patch_area_threshold")}
            except Exception as e:
                print(f"Warning: Could not load U-Net checkpoint: {e}")

        if mask_threshold is not None:
            self.mask_threshold = mask_threshold
        if detection_area_threshold is not None:
            self.detection_area_threshold = detection_area_threshold

    @staticmethod
    def _to_rgb_array(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            return np.array(Image.open(image_input).convert("RGB"))
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        if isinstance(image_input, np.ndarray):
            return image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        raise TypeError("Unsupported image input format.")

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        defect_hint: str = "corrosion",
        run_baseline: bool = True,
        gt_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Run defect prediction on an image.

        Returns a dict with the U-Net probability map, binary mask, boxes, decision and
        confidence, plus (optionally) the baseline's per-tile classification. If gt_mask is
        given, baseline tiles also carry their ground-truth label for scoring.
        """
        img_arr = self._to_rgb_array(image_input)
        orig_h, orig_w = img_arr.shape[:2]

        baseline_res = None
        if run_baseline and self.baseline_model is not None:
            baseline_res = self.baseline_model.predict_patches(
                img_arr, self.patch_config["long_side"], self.patch_config["patch_size"], mask=gt_mask)

        t0 = time.perf_counter()
        if self.unet_model is not None:
            prob_map = predict_probability_map(self.unet_model, img_arr, self.work_long_side, self.device)
            bin_mask = (prob_map >= self.mask_threshold).astype(np.uint8)
            defect_area_frac = float(bin_mask.mean())
            boxes = extract_defect_boxes(bin_mask, defect_type=defect_hint,
                                         min_area=max(30, int(0.001 * orig_h * orig_w)))
            has_defect = defect_area_frac >= self.detection_area_threshold
            if has_defect:
                # Mean defect probability inside the predicted defect region
                confidence = float(prob_map[bin_mask > 0].mean())
            else:
                # Confidence the frame is sound = 1 - mean of the top-1% most suspicious pixels
                top = np.sort(prob_map.ravel())[-max(1, prob_map.size // 100):]
                confidence = float(1.0 - top.mean())
        elif baseline_res is not None:
            has_defect = baseline_res["prediction"] == 1
            probs = [t["defect_probability"] for t in baseline_res["tiles"]]
            confidence = float(max(probs)) if has_defect else float(1.0 - max(probs))
            prob_map = np.zeros((orig_h, orig_w), dtype=np.float32)
            for t in baseline_res["tiles"]:
                prob_map[t["y1"]:t["y2"], t["x1"]:t["x2"]] = np.maximum(
                    prob_map[t["y1"]:t["y2"], t["x1"]:t["x2"]], t["defect_probability"])
            bin_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            boxes = []
            defect_area_frac = 0.0
        else:
            raise RuntimeError("No model is loaded (neither U-Net nor Baseline).")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_defect_class": f"suspected_{defect_hint}" if has_defect else "sound_hull",
            "status": "requires_further_inspection" if has_defect else "clear",
            "confidence_score": round(confidence, 4),
            "defect_detected": bool(has_defect),
            "probability_map": prob_map,
            "binary_mask": bin_mask,
            "bounding_boxes": boxes,
            "defect_area_fraction": round(defect_area_frac, 6),
            "inference_time_ms": round(latency_ms, 2),
            "baseline_result": baseline_res,
            "image_shape": (orig_h, orig_w),
        }

    @staticmethod
    def draw_overlay(
        image: np.ndarray,
        pred_dict: Dict[str, Any],
        show_mask: bool = True,
        show_boxes: bool = True,
        defect_color: Tuple[int, int, int] = (255, 0, 200)
    ) -> np.ndarray:
        """Render the predicted defect mask (magenta) and bounding boxes on the image."""
        vis = image.copy()
        mask = pred_dict.get("binary_mask")
        if show_mask and mask is not None and mask.shape[:2] == vis.shape[:2] and np.sum(mask) > 0:
            m = mask > 0
            vis[m] = (0.45 * vis[m] + 0.55 * np.array(defect_color)).astype(np.uint8)

        if show_boxes:
            thick = max(2, int(round(max(vis.shape[:2]) / 400)))
            for b in pred_dict.get("bounding_boxes", []):
                x1, y1, x2, y2 = b["bbox_pixel"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), defect_color, thick)
        return vis

    @staticmethod
    def draw_error_map(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
        """Colour-code prediction vs ground truth.

        Green = correctly found corrosion (true positive), Red = predicted but not corrosion
        (false positive), Blue = corrosion the model missed (false negative). Sound pixels that
        were correctly left alone are shown greyed out.
        """
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        vis = (gray * 0.55).astype(np.uint8)
        p, g = pred_mask > 0, gt_mask > 0
        vis[p & g] = (0, 200, 70)
        vis[p & ~g] = (235, 40, 40)
        vis[~p & g] = (40, 110, 255)
        return vis
