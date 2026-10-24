# 🌊 AI-Powered Underwater Ship Hull Inspection Drone

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.8%2Bcu129-red.svg)](https://pytorch.org/)
[![Apache Arrow](https://img.shields.io/badge/Apache%20Arrow-21.0-orange.svg)](https://arrow.apache.org/)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/PyTest-28%20Passed-brightgreen.svg)](https://pytest.org/)

An end-to-end, academic-grade **Fundamentals of Machine Learning (FOML)** project simulating an autonomous underwater remotely operated vehicle (ROV) inspecting commercial ship hulls for visible structural defects (corrosion scale, oxidation, and fatigue cracks).

The system integrates **traditional machine learning (handcrafted colour/texture features + SVM)**, **deep convolutional vision (Residual U-Net)**, **robotics trajectory and optical physics simulation**, **columnar Apache Parquet telemetry logging via PyArrow**, and an **interactive multi-page Streamlit inspection dashboard** with automated PDF report generation.

---

## 📌 Key System Features

1. **Dual Machine Learning Pipeline**:
   - **Model A (Traditional ML Baseline)**: 47 handcrafted features per 256 px tile (Lab colour relative to the whole image, LBP texture, gradient statistics) + RBF Support Vector Machine. Hyperparameters chosen by scene-grouped cross-validation with the one-standard-error rule.
   - **Model B (Deep Learning U-Net)**: Compact residual U-Net (2.0M parameters) predicting a corrosion mask for every pixel, trained on random crops with heavy augmentation and early stopping. Decision thresholds are calibrated on the validation split.
   - **Loss Formulation for Severe Class Imbalance**: Combined Binary Cross-Entropy + Soft Dice Loss ($\alpha=1.0, \beta=1.2, \gamma=0.5$).

2. **Underwater Robotics & Optical Physics Simulation**:
   - **Parametric Hull Geometry**: 50.0m $\times$ 10.0m vessel side surface partitioned into 10 structural inspection zones (Bow, Midship, Stern, Keel, Waterline).
   - **ROV Kinematics**: Serpentine (lawnmower) path planner, thruster dynamics, acoustic DVL/USBL positioning uncertainty ($\sigma_{\text{pos}}=0.04$m), and coverage grid tracking.
   - **Jaffe-McGlamery Optical Model**: Wavelength-dependent red light absorption ($c=0.65$), forward-scattering turbidity blur, contrast attenuation, and suspended marine snow particles.

3. **High-Performance Columnar Storage**:
   - Strongly-typed Apache Arrow schema capturing 28 telemetry and inspection attributes per observation step.
   - Parquet serialization with Snappy compression for sub-millisecond analytical queries and mission replay.

4. **Interactive Streamlit Dashboard** (grouped sidebar navigation, light and dark theme):
   - **Home** &mdash; Headline results and a 3-step tour: overfitting check, inspect a test image, fly the drone.
   - **Project overview** &mdash; Datasets, the scene-grouped split and the two models' held-out results.
   - **Image inspection** &mdash; Pick a TEST / VAL / TRAIN image (or upload one); see ground truth vs U-Net mask, a colour error map, a good/fair/poor verdict and the SVM's per-tile decisions.
   - **Drone simulation** &mdash; ROV camera with HUD, hull map, sensors and a live scorecard that checks every frame against its ground truth.
   - **Model evaluation** &mdash; Overfitting check with confidence intervals, learning curves, Model A vs B, segmentation, robustness and limitations.
   - **Dataset explorer** &mdash; Image/mask browser with split labels and a leakage check.
   - **Inspection reports** &mdash; Parquet mission logs with CSV and PDF export.

---

## 📊 Experimental Benchmark Results

All numbers below are produced by `scripts/evaluate_model.py` and saved in `outputs/metrics/test_metrics_mavecodd.json`.

### Evaluation protocol (why these numbers can be trusted)
- **Scene-grouped split, 60/20/20.** Consecutive frame numbers are near-duplicates (video / photo bursts), so they are grouped into *scenes* of 5 frame IDs and each scene goes to exactly one split, stratified by HiRes/LoRes. Result: 44 train / 16 validation / 16 test images, no scene shared between splits.
- **Same task for both models.** Every image is rescaled to 1024 px and tiled into 256 px tiles; a tile is *defect* if ≥ 5% of it is corrosion and *sound* if ≤ 0.5% (ambiguous tiles are skipped). Both models are scored on exactly these tiles.
- **No image-level "defect yes/no" score.** Every MaVeCoDD photo contains corrosion, so that score would be 100% for a model that always answers "defect". (An earlier version of this README reported exactly such a 100% result; it measured nothing.)
- **No test-set tuning.** Hyperparameters, early stopping and both U-Net thresholds use training/validation data only. Test confidence intervals come from a 1,000-sample bootstrap over whole test images.

### Overfitting check (train vs held-out)

| Metric | Train | Validation | Test (95% CI) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Model A · tile balanced accuracy | 0.909 | 0.906 | 0.732 (0.58 – 0.86) | Mild gap, appears only on the test scenes |
| Model B · tile balanced accuracy | 0.905 | 0.938 | 0.775 (0.64 – 0.89) | No meaningful overfitting |
| Model B · segmentation Dice | 0.637 | 0.633 | 0.655 (0.57 – 0.74) | No meaningful overfitting |

The U-Net's train and validation Dice stayed within ±0.02 of each other during all 34 training epochs (learning curves on the *Model Evaluation* page). Both models score lower on the test tiles than on validation; the test split contains only **21 clearly sound tiles**, so one tile moves balanced accuracy by ~2.4 points and the intervals are wide.

### Model A vs Model B on the same 165 held-out test tiles (144 defect / 21 sound)

| Metric | Model A: features + SVM | Model B: U-Net |
| :--- | :---: | :---: |
| Balanced accuracy | 0.732 | **0.775** |
| Accuracy | 0.709 | **0.927** |
| Precision (defect) | **0.953** | 0.940 |
| Recall (defect) | 0.701 | **0.979** |
| F1 (defect) | 0.808 | **0.959** |
| ROC AUC | **0.815** | 0.776 |
| Missed defect tiles (FN) | 43 | **3** |
| False alarms (FP) | **5** | 9 |
| Latency (CPU) | 35 ms / tile | 311 ms / full image |
| Checkpoint size | 0.13 MB | 8.2 MB |

**U-Net pixel segmentation (test):** Dice 0.655, IoU 0.510, pixel recall 86%, pixel precision 57%. The U-Net misses very little corrosion but over-segments: most of its false alarms are deck railings / sky at the top of dry-dock photos and weathered coating that the annotators did not mark (e.g. `hull005`, Dice 0.28).

### Optical turbidity stress test (test split, fixed seeds)

| Turbidity | Condition | U-Net Dice | U-Net tile bal. acc | SVM tile bal. acc |
| :---: | :--- | :---: | :---: | :---: |
| 0.00 | Clear Water | 0.655 | 0.775 | 0.732 |
| 0.15 | Light Coastal | 0.538 | 0.758 | 0.723 |
| 0.30 | Harbour Waters | 0.525 | 0.711 | 0.706 |
| 0.45 | Turbid Estuarine | 0.510 | 0.714 | 0.620 |
| 0.60 | Heavy Turbidity | 0.491 | 0.667 | 0.625 |

### Cross-domain generalisation
Zero-shot on 40 Underwater Crack test images the corrosion U-Net reaches **Dice 0.066 / IoU 0.038**: a dry-dock corrosion model does not transfer to thin underwater cracks without target-domain data.

### Simulation scorecard
`scripts/run_simulation.py --steps 400` scores every camera frame against its ground-truth mask. Frames are cut from held-out **test** photos, restricted to windows that show hull plate (no sky/railings). At turbidity 0.2: 46/46 corroded frames detected, 0/354 false alarms; at turbidity 0.5: 45/46 detected, 0 false alarms. This is an easy setting (plain paint vs heavy corrosion); the tile-level numbers above are the representative ones.

---

## 🏗️ Repository Architecture

```
FOML_proj/
│
├── config/
│   └── config.yaml              # Central configuration parameters
├── data/
│   ├── raw/
│   │   ├── mavecodd/            # Primary Marine Vessel Corrosion dataset
│   │   └── underwater_crack/    # Secondary Underwater Crack dataset
│   ├── processed/               # Stratified leakage-free split manifests
│   └── sample/                  # Quick-start synthetic hull samples
├── models/
│   └── checkpoints/             # Trained .joblib and .pth model weights
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_cnn_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── annotation_parser.py # Mask decoding and bbox extraction
│   │   ├── preprocessing.py     # Degradation simulation and HOG features
│   │   ├── split_dataset.py     # Scene-grouped leakage-free splits
│   │   └── dataset_loader.py    # PyTorch dataset & sample generators
│   ├── models/
│   │   ├── baseline.py          # Model A: HOG+Color+PCA+SVM
│   │   ├── cnn.py               # Model B: Multi-task Residual U-Net
│   │   ├── train.py             # Optimizer, scheduler, and early stopping
│   │   ├── predict.py           # Unified inference and visual overlay
│   │   └── evaluate.py          # Benchmark metrics and stress testing
│   ├── simulation/
│   │   ├── hull.py              # 50m x 10m hull geometry and coverage grid
│   │   ├── rov.py               # Kinematics, thrusters, and sensor noise
│   │   ├── inspection_path.py   # Serpentine waypoint path planner
│   │   ├── camera.py            # Optical footprint and degradation
│   │   └── simulator.py         # Master simulation orchestration loop
│   ├── storage/
│   │   ├── schema.py            # Strongly typed 28-field PyArrow schema
│   │   ├── parquet_writer.py    # Columnar Parquet logging and batch append
│   │   └── parquet_reader.py    # Analytical queries and CSV export
│   ├── reporting/
│   │   └── report_generator.py  # Publication-grade PDF reports (ReportLab)
│   └── utils/
│       ├── config.py            # Config loader with environment overrides
│       └── reproducibility.py   # Deterministic random seeds and device selector
├── app/
│   ├── streamlit_app.py         # Entry point: grouped navigation + sidebar status
│   ├── ui.py                    # Shared theme-safe styling and metric loading
│   └── pages/
│       ├── 0_Home.py
│       ├── 1_Overview.py
│       ├── 2_Image_Inspection.py
│       ├── 3_Drone_Simulation.py
│       ├── 4_Dataset_Explorer.py
│       ├── 5_Model_Evaluation.py
│       └── 6_Inspection_Reports.py
├── scripts/
│   ├── download_datasets.py     # Download both public datasets (SHA-256 verified)
│   ├── prepare_dataset.py       # Extraction and split manifests
│   ├── inspect_dataset.py       # Dataset statistics and shape verification
│   ├── train_model.py           # Model training runner
│   ├── evaluate_model.py        # Benchmark and stress test runner
│   └── run_simulation.py        # Autonomous ROV simulation runner
├── outputs/
│   ├── figures/                 # Confusion matrices and response curves
│   ├── metrics/                 # JSON test metrics summaries
│   └── reports/                 # Compiled academic report (PDF & MD)
├── tests/                       # PyTest test suite (28 tests)
├── requirements.txt             # Pinned project dependencies
└── README.md
```

---

## 🚀 Quick-Start Guide

### 1. Environment Installation
Ensure Python 3.11+ is installed. Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd FOML_proj
pip install -r requirements.txt
```

### 2. Dataset Download & Preparation
Both datasets are public on Mendeley Data (CC BY 4.0). Download them (checksums are verified), then build the scene-grouped split manifests:
```bash
python scripts/download_datasets.py
python scripts/prepare_dataset.py
```
*Without the datasets, `prepare_dataset.py` still creates a small synthetic sample set in `data/sample/` so the pipeline can be smoke-tested.*

### 3. Model Training
```bash
# Full training on MaVeCoDD (~1 min for the SVM, ~35 min for the U-Net on a laptop CPU):
python scripts/train_model.py --dataset mavecodd

# Retrain only one model:
python scripts/train_model.py --dataset mavecodd --only baseline

# Quick smoke test on the synthetic samples:
python scripts/train_model.py --dataset sample --epochs 2
```

### 4. Model Evaluation & Benchmark Analysis
Evaluate both models on train / validation / test (overfitting check), then run the turbidity and cross-domain experiments:
```bash
python scripts/evaluate_model.py --dataset mavecodd
```

### 5. Autonomous Drone Inspection Simulation
Run an autonomous serpentine inspection run logging telemetry to Apache Parquet:
```bash
python scripts/run_simulation.py --steps 120 --turbidity 0.25
```

### 6. Run Test Suite
Verify all components with automated unit and integration tests:
```bash
pytest tests/ -v
```

### 7. Launch Interactive Streamlit Dashboard
Launch the web interface in your browser:
```bash
streamlit run app/streamlit_app.py
```

---

## ⚠️ Scientific & Marine Engineering Limitations

1. **Optical vs. Ultrasonic Testing**: Computer vision analyzes visible surface phenomena (oxide scale, coating failure, surface fissure cracks). It **cannot determine internal metallurgical stress, plate thinning, or subsurface fatigue without ultrasonic testing (UT) or eddy-current sensors**.
2. **Defect Nomenclature**: Under classification society rules (DNV, Lloyd's Register), all vision detections are cataloged as **'suspected defects'** with status **'requires further inspection'**. Predictions are not certified proof of structural failure.
3. **Small test set**: 16 test images (21 clearly sound tiles) give wide confidence intervals; treat differences of a few points between the models as noise.
4. **Sensor Noise**: Real-world acoustic positioning (USBL/DVL) exhibits multipath reflections and drift; simulated sensor noise demonstrates the need for Kalman filtering in physical deployments.

---

## 📜 Citation & Credits
- **Primary Dataset**: Kamzelis, K., Chliveros, G., & Tzanetatos, I. (2021). *MaVeCoDD: Marine Vessel Corrosion Detection Dataset in Dry-Dock Images*. Mendeley Data, V1, doi: [10.17632/ry392rp8cj.1](https://doi.org/10.17632/ry392rp8cj.1).
- **Secondary Dataset**: Gao, Y., et al. (2021). *Underwater Crack Image Dataset for Structural Health Monitoring*. Mendeley Data, V1, doi: [10.17632/g962rbm77b/1](https://doi.org/10.17632/g962rbm77b/1).
- **Optical Model**: Jaffe, J. S. (1990). *Computer modeling and the design of optimal underwater imaging systems*. IEEE JOE.
