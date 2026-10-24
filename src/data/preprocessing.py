"""Preprocessing and underwater degradation pipeline."""

from typing import Tuple, Optional
import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def resize_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize image and optional binary mask to target dimensions.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Optional binary mask array of shape (H, W).
        target_size: Desired (width, height).

    Returns:
        Resized image and resized mask.
    """
    w, h = target_size
    resized_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    if mask is not None:
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 0).astype(np.uint8)
    else:
        resized_mask = None

    return resized_img, resized_mask


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_long_side(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    long_side: int = 512
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize so the longer image side equals long_side, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = long_side / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    resized_mask = None
    if mask is not None:
        resized_mask = (cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return resized, resized_mask


def pad_to_multiple(image: np.ndarray, multiple: int = 16) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Reflect-pad bottom/right so both sides are divisible by `multiple` (U-Net requirement).

    Returns the padded image and the original (h, w) for cropping the output back.
    """
    h, w = image.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return image, (h, w)


def to_normalized_tensor_array(image: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3xHxW normalized with ImageNet statistics."""
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose((2, 0, 1)).copy()


def tile_positions(length: int, patch_size: int) -> list:
    """Evenly spaced tile start offsets covering [0, length) with minimal overlap."""
    if length <= patch_size:
        return [0]
    n = int(np.ceil(length / patch_size))
    return [int(round(v)) for v in np.linspace(0, length - patch_size, n)]


def label_patch(defect_frac: float, defect_min_frac: float = 0.05, sound_max_frac: float = 0.005) -> int:
    """1 = defect tile, 0 = sound tile, -1 = ambiguous (excluded from patch metrics)."""
    if defect_frac >= defect_min_frac:
        return 1
    if defect_frac <= sound_max_frac:
        return 0
    return -1


def extract_patch_grid(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    long_side: int = 1024,
    patch_size: int = 256,
    defect_min_frac: float = 0.05,
    sound_max_frac: float = 0.005
) -> list:
    """Rescale image and tile it into a deterministic grid of patches.

    This single definition of a 'patch' is shared by the baseline training, the
    evaluation of both models and the dashboard, so all patch metrics are comparable.

    Returns:
        List of dicts with keys: image, x, y, size, scale, defect_frac, label.
        x/y/size are in the rescaled image; divide by `scale` for original pixels.
    """
    h0, w0 = image.shape[:2]
    img, msk = resize_long_side(image, mask, long_side)
    scale = img.shape[1] / float(w0)
    if min(img.shape[:2]) < patch_size:
        pad_h = max(0, patch_size - img.shape[0])
        pad_w = max(0, patch_size - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if msk is not None:
            msk = cv2.copyMakeBorder(msk, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    context = image_color_context(img)
    h, w = img.shape[:2]
    patches = []
    for y in tile_positions(h, patch_size):
        for x in tile_positions(w, patch_size):
            frac = float(msk[y:y + patch_size, x:x + patch_size].mean()) if msk is not None else float("nan")
            patches.append({
                "image": img[y:y + patch_size, x:x + patch_size],
                "x": x,
                "y": y,
                "size": patch_size,
                "scale": scale,
                "defect_frac": frac,
                "label": label_patch(frac, defect_min_frac, sound_max_frac) if msk is not None else -1,
                "context": context,
            })
    return patches


def apply_underwater_degradation(
    image: np.ndarray,
    turbidity: float = 0.3,
    red_attenuation: float = 0.65,
    contrast_factor: float = 0.8,
    marine_snow_density: float = 0.001,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Simulate underwater optical physics (Jaffe-McGlamery optical model approximation).

    Simulates:
    1. Selective spectral absorption (red wavelengths attenuate rapidly in water).
    2. Forward scattering / turbidity (haze and blurring).
    3. Contrast loss due to veiling light / backscatter.
    4. Marine snow / suspended particulate matter.

    Args:
        image: RGB image in uint8 [0, 255].
        turbidity: Scattering coefficient in [0.0, 1.0].
        red_attenuation: Factor to suppress red channel [0.0, 1.0].
        contrast_factor: Contrast retention [0.0, 1.0].
        marine_snow_density: Density of suspended particle specks.
        random_seed: Optional seed for reproducibility.

    Returns:
        Degraded RGB image in uint8 [0, 255].
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    img_float = image.astype(np.float32) / 255.0

    # 1. Wavelength-dependent absorption (Red drops heavily, Green/Blue preserved)
    # Channel order: RGB
    img_float[:, :, 0] *= (1.0 - red_attenuation * 0.7)  # Red channel
    img_float[:, :, 1] *= (1.0 - red_attenuation * 0.15) # Green channel
    img_float[:, :, 2] *= 1.0                            # Blue channel

    # 2. Add greenish-blue backscatter veiling light
    veiling_light = np.array([0.05, 0.25, 0.35], dtype=np.float32) * turbidity
    img_float = img_float * (1.0 - turbidity * 0.5) + veiling_light

    # 3. Contrast attenuation
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = mean_val + contrast_factor * (img_float - mean_val)

    # 4. Turbidity / forward-scattering blur
    if turbidity > 0.1:
        kernel_size = int(round(turbidity * 6)) * 2 + 1
        if kernel_size >= 3:
            img_float = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigmaX=turbidity * 2.0)

    # 5. Marine snow / particulate noise
    if marine_snow_density > 0:
        h, w = image.shape[:2]
        num_particles = int(h * w * marine_snow_density)
        if num_particles > 0:
            ys = rng.integers(0, h, num_particles)
            xs = rng.integers(0, w, num_particles)
            particle_brightness = rng.uniform(0.6, 1.0, num_particles)
            for y, x, b in zip(ys, xs, particle_brightness):
                radius = rng.integers(1, 3)
                cv2.circle(img_float, (int(x), int(y)), radius, (float(b), float(b), float(b)), -1)

    degraded = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    return degraded


def image_color_context(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median and spread of the whole image in CIE-Lab colour space.

    Patch colours are expressed relative to this, which removes the overall paint colour of a
    particular vessel / lighting condition (the main source of scene-to-scene variation).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0), lab.std(axis=0) + 1e-3


def extract_handcrafted_features(
    image: np.ndarray,
    context: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """Compact colour + texture descriptor used by Model A (47 values).

    1. Lab colour of the patch relative to the whole image (mean, std, 10th/90th percentile and
       the share of strongly deviating pixels per channel)            -> 15
    2. Uniform LBP texture histograms at two radii (8,1) and (16,3)   -> 28
    3. Gradient-magnitude statistics (rough / flaking surfaces)       -> 4

    Scene-grouped cross-validation showed this generalises to unseen scenes far better than
    large HOG descriptors (balanced accuracy ~0.93 vs ~0.78), whose 8000+ dimensions mostly
    captured scene-specific texture.

    Args:
        image: RGB uint8 patch.
        context: Output of image_color_context() for the image the patch was cut from.
            If None, the patch itself is used as context.
    """
    if context is None:
        context = image_color_context(image)
    med, spread = context

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    z = (lab - med) / spread
    color = np.concatenate([z.mean(0), z.std(0), np.percentile(z, 10, axis=0),
                            np.percentile(z, 90, axis=0), (np.abs(z) > 1.5).mean(0)])

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = []
    for n_points, radius in ((8, 1), (16, 3)):
        lbp = local_binary_pattern(gray, n_points, radius, "uniform")
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        textures.append(hist)

    g = gray.astype(np.float32)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    grad = np.array([mag.mean(), mag.std(), np.percentile(mag, 90), (mag > 40).mean()])

    return np.concatenate([color, *textures, grad]).astype(np.float32)
