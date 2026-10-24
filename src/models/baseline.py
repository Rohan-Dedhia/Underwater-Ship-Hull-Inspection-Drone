"""Traditional ML Baseline: handcrafted colour/texture features + Support Vector Machine."""

import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from src.data.preprocessing import extract_handcrafted_features, extract_patch_grid
from src.data.dataset_loader import load_image_and_mask


class HullDefectBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Model A: handcrafted colour/texture features -> standardisation -> (optional PCA) -> RBF-SVM."""

    def __init__(
        self,
        use_pca: bool = False,
        pca_components: int = 16,
        svm_c: float = 1.0,
        svm_gamma: str = "scale",
        svm_kernel: str = "rbf",
        random_state: int = 42
    ):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state) if use_pca else None
        self.classifier = SVC(
            C=svm_c,
            gamma=svm_gamma,
            kernel=svm_kernel,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def extract_features(self, image: np.ndarray, context=None) -> np.ndarray:
        """Feature vector of one RGB patch; `context` = image_color_context() of its source image."""
        return extract_handcrafted_features(image, context)

    def extract_batch_features(self, images: List[np.ndarray], contexts: Optional[List] = None) -> np.ndarray:
        """Extract features for a list of RGB patches (optionally with their image contexts)."""
        contexts = contexts if contexts is not None else [None] * len(images)
        return np.vstack([self.extract_features(img, ctx) for img, ctx in zip(images, contexts)])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled) if self.use_pca and self.pca is not None else X_scaled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HullDefectBaselineClassifier":
        """Fit scaler, optional PCA, and SVM on extracted feature matrix X and labels y."""
        X_scaled = self.scaler.fit_transform(X)
        if self.use_pca and self.pca is not None:
            max_comp = min(X.shape[0] - 1, X.shape[1], self.pca_components)
            if max_comp != self.pca.n_components:
                self.pca = PCA(n_components=max_comp, random_state=self.random_state)
            X_trans = self.pca.fit_transform(X_scaled)
        else:
            X_trans = X_scaled
        self.classifier.fit(X_trans, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary defect labels (0: Sound Hull, 1: Suspected Defect)."""
        return self.classifier.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Defect score in [0, 1]: logistic squash of the SVM decision value (0.5 = decision boundary).

        This is a ranking score (used for ROC AUC and display), not a calibrated probability.
        """
        d = self.classifier.decision_function(self._transform(X))
        p = 1.0 / (1.0 + np.exp(-2.0 * d))
        return np.column_stack([1.0 - p, p])

    def predict_patches(
        self,
        image: np.ndarray,
        long_side: int = 1024,
        patch_size: int = 256,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Classify every tile of the shared patch grid (the task this model was trained on).

        Tile coordinates are returned in ORIGINAL image pixels. If a ground-truth mask is
        given, each tile also carries its true label (-1 = ambiguous, not scored).
        """
        t0 = time.perf_counter()
        patches = extract_patch_grid(image, mask, long_side, patch_size)
        X = self.extract_batch_features([p["image"] for p in patches], [p["context"] for p in patches])
        preds = self.predict(X)
        probs = self.predict_proba(X)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        tiles = []
        for p, pred, prob in zip(patches, preds, probs):
            sc = p["scale"]
            tiles.append({
                "x1": int(p["x"] / sc), "y1": int(p["y"] / sc),
                "x2": int((p["x"] + p["size"]) / sc), "y2": int((p["y"] + p["size"]) / sc),
                "prediction": int(pred), "defect_probability": float(prob),
                "true_label": int(p["label"]), "true_defect_frac": p["defect_frac"],
            })
        n_def = sum(t["prediction"] for t in tiles)
        return {
            "tiles": tiles,
            "defect_tiles": int(n_def),
            "total_tiles": len(tiles),
            "prediction": int(n_def > 0),
            "defect_class": "suspected_defect" if n_def > 0 else "sound_hull",
            "latency_ms": round(latency_ms, 2),
        }

    def save(self, filepath: str | Path) -> None:
        """Save model bundle to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "HullDefectBaselineClassifier":
        """Load model bundle from disk."""
        return joblib.load(filepath)


def prepare_patch_dataset(
    samples: List[Dict[str, Any]],
    long_side: int = 1024,
    patch_size: int = 256,
    defect_min_frac: float = 0.05,
    sound_max_frac: float = 0.005,
    return_meta: bool = False
):
    """Tile every scene into the shared deterministic patch grid and keep unambiguous tiles.

    Labels: 1 = >= defect_min_frac corrosion pixels, 0 = <= sound_max_frac. Tiles in between
    are skipped so the label is never decided by a sliver of corrosion at a tile edge.

    Returns:
        (images, labels) or, with return_meta=True, (images, labels, meta) where meta holds
        the scene_id (for grouped cross-validation), image_id and tile position per patch.
    """
    images_list: List[np.ndarray] = []
    labels_list: List[int] = []
    meta: List[Dict[str, Any]] = []

    for s in samples:
        img, mask = load_image_and_mask(s)
        for p in extract_patch_grid(img, mask, long_side, patch_size, defect_min_frac, sound_max_frac):
            if p["label"] < 0:
                continue
            images_list.append(p["image"])
            labels_list.append(p["label"])
            meta.append({
                "context": p["context"],
                "scene_id": s.get("scene_id", s.get("image_id")),
                "image_id": s.get("image_id"),
                "x": p["x"], "y": p["y"], "defect_frac": p["defect_frac"],
            })

    labels = np.array(labels_list, dtype=int)
    if return_meta:
        return images_list, labels, meta
    return images_list, labels


def select_hyperparameters_by_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[str],
    c_values: List[float],
    pca_values: List[int],
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Scene-grouped K-fold CV over (C, PCA components) with the one-standard-error rule.

    Patches from one scene never appear in both the CV-train and CV-validation fold.
    Among all settings whose mean balanced accuracy is within one standard error of the best,
    the simplest one (no / fewest PCA components, then smallest C = strongest regularisation) is
    chosen. This deliberately trades a negligible amount of CV score for less overfitting.
    """
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(n_folds, n_groups))
    results = []
    for n_comp in pca_values:
        for c in c_values:
            steps = [("scaler", StandardScaler())]
            if n_comp:
                steps.append(("pca", PCA(n_components=n_comp, random_state=random_state)))
            steps.append(("svm", SVC(C=c, gamma="scale", kernel="rbf", class_weight="balanced")))
            pipe = Pipeline(steps)
            out = cross_validate(pipe, X, y, groups=groups, cv=cv,
                                 scoring="balanced_accuracy", return_train_score=True)
            results.append({
                "C": c,
                "pca_components": n_comp,
                "cv_mean": float(np.mean(out["test_score"])),
                "cv_std": float(np.std(out["test_score"])),
                "cv_se": float(np.std(out["test_score"]) / np.sqrt(len(out["test_score"]))),
                "train_mean": float(np.mean(out["train_score"])),
            })

    best = max(results, key=lambda r: r["cv_mean"])
    eligible = [r for r in results if r["cv_mean"] >= best["cv_mean"] - best["cv_se"]]
    chosen = sorted(eligible, key=lambda r: (r["pca_components"] or 0, r["C"]))[0]
    return {"chosen": chosen, "best": best, "grid": results, "n_folds": cv.n_splits}
