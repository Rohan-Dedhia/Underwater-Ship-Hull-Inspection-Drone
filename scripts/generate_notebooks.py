"""Script to generate educational Jupyter Notebooks for the FOML project."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]}


def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.strip().split("\n")]}


# 1. Notebook 01: Dataset Exploration
nb1_cells = [
    md_cell("# Notebook 01: Underwater Hull Dataset Exploration & Annotation Parsing\n\n"
            "This notebook inspects the primary **MaVeCoDD** (Marine Vessel Corrosion) dataset and the secondary **Underwater Crack** dataset. "
            "We verify image dimensions, parse pixel-level masks, extract bounding boxes, and formulate leakage-free splits."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "from src.utils.config import load_config\n"
              "from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask\n"
              "from src.data.annotation_parser import parse_mavecodd_mask, extract_defect_boxes\n"
              "import matplotlib.pyplot as plt\nimport numpy as np\n\n"
              "config = load_config()\nprint('Loaded configuration:', config['project']['name'])"),
    md_cell("## 1. Inspect MaVeCoDD Samples\n\nLet's scan raw images and mask pairs from MaVeCoDD."),
    code_cell("mav_samples = scan_mavecodd_samples('../data/raw/mavecodd', zip_fallback='../ry392rp8cj-1.zip')\n"
              "print(f'Total matched MaVeCoDD pairs: {len(mav_samples)}')\n\n"
              "img, mask = load_image_and_mask(mav_samples[0])\n"
              "print('Sample shape:', img.shape, 'Mask shape:', mask.shape)\n\n"
              "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
              "axes[0].imshow(img)\naxes[0].set_title('Raw Marine Vessel Hull')\naxes[0].axis('off')\n"
              "axes[1].imshow(mask, cmap='hot')\naxes[1].set_title('Parsed Binary Corrosion Mask')\naxes[1].axis('off')\n"
              "plt.tight_layout()\nplt.show()"),
    md_cell("## 2. Bounding Box Extraction via Connected Components\n\n"
            "We can derive object-detection bounding boxes from the segmentation masks without manual box labeling."),
    code_cell("boxes = extract_defect_boxes(mask, defect_type='corrosion')\n"
              "print(f'Extracted {len(boxes)} defect bounding boxes')\n"
              "for b in boxes[:3]:\n    print(b)")
]

with open(NB_DIR / "01_dataset_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb1_cells), f, indent=2)

# 2. Notebook 02: Baseline Model
nb2_cells = [
    md_cell("# Notebook 02: Traditional ML Baseline (handcrafted features + SVM)\n\n"
            "**Model A** classifies 256 px tiles as *defect* (>= 5% corrosion) or *sound* (<= 0.5%). "
            "Features: Lab colour relative to the whole image, LBP texture and gradient statistics (47 values). "
            "Hyperparameters are chosen by **scene-grouped cross-validation** on the training split, "
            "using the one-standard-error rule to prefer the most regularised model."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "import numpy as np\n"
              "from src.utils.config import load_config\n"
              "from src.data.split_dataset import load_split_manifest\n"
              "from src.models.baseline import (HullDefectBaselineClassifier, prepare_patch_dataset,\n"
              "                                 select_hyperparameters_by_group_cv)\n"
              "from src.models.evaluate import evaluate_baseline_on_patches\n\n"
              "config = load_config()\nsplits = load_split_manifest('../data/processed/split_manifest_mavecodd.json')\n"
              "patch_cfg = {'long_side': 1024, 'patch_size': 256, 'defect_min_frac': 0.05, 'sound_max_frac': 0.005}\n"
              "print({k: len(v) for k, v in splits.items()})"),
    md_cell("## 1. Patches, features and grouped cross-validation"),
    code_cell("train_imgs, train_y, meta = prepare_patch_dataset(splits['train'], return_meta=True, **patch_cfg)\n"
              "print(f'Training patches: {len(train_imgs)} (defect {train_y.sum()}, sound {(train_y == 0).sum()})')\n\n"
              "X_train = HullDefectBaselineClassifier().extract_batch_features(train_imgs, [m['context'] for m in meta])\n"
              "cv = select_hyperparameters_by_group_cv(X_train, train_y, [m['scene_id'] for m in meta],\n"
              "                                        c_values=[0.03, 0.1, 0.3, 1, 3, 10], pca_values=[None, 8, 16])\n"
              "print('Chosen:', cv['chosen'])\n\n"
              "model = HullDefectBaselineClassifier(use_pca=cv['chosen']['pca_components'] is not None,\n"
              "                                     pca_components=cv['chosen']['pca_components'] or 16,\n"
              "                                     svm_c=cv['chosen']['C']).fit(X_train, train_y)"),
    md_cell("## 2. Train vs validation vs test (overfitting check)"),
    code_cell("for split in ['train', 'val', 'test']:\n"
              "    r = evaluate_baseline_on_patches(model, splits[split], patch_cfg)\n"
              "    print(f\"{split:5s} balanced acc {r['balanced_accuracy']:.3f}  recall {r['recall']:.3f}  \"\n"
              "          f\"precision {r['precision']:.3f}  {r['confusion_matrix']}\")")
]

with open(NB_DIR / "02_baseline_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

# 3. Notebook 03: CNN Training
nb3_cells = [
    md_cell("# Notebook 03: Deep Learning U-Net Training & Segmentation\n\n"
            "**Model B** is a compact U-Net trained on random 256 px crops (zoom, flips, rotations, colour jitter, "
            "simulated turbidity) with a BCE + soft Dice loss. Train and validation Dice are measured on full images "
            "every epoch, and early stopping keeps the epoch with the lowest validation loss."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "import json\nimport matplotlib.pyplot as plt\n"
              "from src.utils.config import load_config\n"
              "from src.utils.reproducibility import get_device\n"
              "from src.data.split_dataset import load_split_manifest\n"
              "from src.models.cnn import HullUNet\n\n"
              "config = load_config()\ndev = get_device('auto')\nprint('Using device:', dev)\n"
              "splits = load_split_manifest('../data/processed/split_manifest_mavecodd.json')"),
    md_cell("## 1. Model size"),
    code_cell("unet = HullUNet(base_channels=config['cnn_model']['base_channels'])\n"
              "print(f\"Trainable parameters: {sum(p.numel() for p in unet.parameters()):,}\")"),
    md_cell("## 2. Learning curves of the trained model\n\n"
            "Full training takes ~40 min on CPU: `python scripts/train_model.py --dataset mavecodd`. "
            "To retrain here instead, use `src.models.train.train_unet(splits['train'], splits['val'], config=config)`."),
    code_cell("summary = json.load(open('../outputs/metrics/training_summary_mavecodd.json'))\n"
              "h = summary['unet']['history']\n"
              "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
              "ax[0].plot(h['epoch'], h['train_loss'], 'o-', label='train'); ax[0].plot(h['epoch'], h['val_loss'], 's-', label='val')\n"
              "ax[0].set_title('Loss'); ax[0].legend()\n"
              "ax[1].plot(h['epoch'], h['train_dice'], 'o-', label='train'); ax[1].plot(h['epoch'], h['val_dice'], 's-', label='val')\n"
              "ax[1].set_title('Dice'); ax[1].set_ylim(0, 1); ax[1].legend()\nplt.show()\n"
              "cal = summary['unet']['calibration']\n"
              "print('Kept epoch:', summary['unet']['best_epoch'], '| pixel threshold:', cal['mask_threshold'],\n"
              "      '| area threshold:', cal['patch_area_threshold'])")
]

with open(NB_DIR / "03_cnn_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb3_cells), f, indent=2)

# 4. Notebook 04: Model Evaluation
nb4_cells = [
    md_cell("# Notebook 04: Comparative Model Evaluation\n\n"
            "Results produced by `python scripts/evaluate_model.py --dataset mavecodd`: both models on the same "
            "test tiles, train/val/test overfitting check, turbidity stress test and cross-domain transfer."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
              "with open('../outputs/metrics/test_metrics_mavecodd.json', 'r') as f:\n"
              "    metrics = json.load(f)\n"
              "print(metrics['evaluation_protocol']['split_sizes'])"),
    md_cell("## 1. Overfitting check (train vs held-out)"),
    code_cell("pd.DataFrame(metrics['overfitting_check']).T"),
    md_cell("## 2. Model A vs Model B on the same test tiles"),
    code_cell("base = metrics['baseline_model_a']['test']\nunet = metrics['deep_learning_model_b']['test']['patch_classification']\n"
              "keys = ['balanced_accuracy', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']\n"
              "pd.DataFrame({'Model A (features+SVM)': [base[k] for k in keys], 'Model B (U-Net)': [unet[k] for k in keys]}, index=keys)"),
    md_cell("## 3. Optical turbidity stress test"),
    code_cell("deg = pd.DataFrame(metrics['optical_degradation_stress_test'])\n"
              "plt.figure(figsize=(8, 4.5))\n"
              "plt.plot(deg['turbidity'], deg['unet_mean_dice'], marker='o', label='U-Net Dice')\n"
              "plt.plot(deg['turbidity'], deg['unet_patch_balanced_accuracy'], marker='s', label='U-Net tile balanced acc')\n"
              "plt.plot(deg['turbidity'], deg['baseline_patch_balanced_accuracy'], marker='^', label='SVM tile balanced acc')\n"
              "plt.xlabel('Turbidity')\nplt.ylim(0, 1.05)\nplt.legend()\nplt.grid(True)\nplt.show()"),
    md_cell("## 4. Cross-domain generalisation to underwater cracks"),
    code_cell("metrics['cross_domain_generalization']")
]

with open(NB_DIR / "04_model_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb4_cells), f, indent=2)

print("All 4 Jupyter Notebooks generated successfully!")
