"""Simulation runner script executing autonomous ROV inspection and PyArrow Parquet logging."""

import os
import sys
import argparse
from pathlib import Path
import json

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device
from src.data.split_dataset import load_split_manifest
from src.simulation.hull import ShipHullSurface
from src.simulation.rov import UnderwaterROV
from src.simulation.camera import SimulatedUnderwaterCamera
from src.simulation.simulator import HullInspectionSimulator
from src.models.predict import HullInspectionPredictor
from src.storage.parquet_writer import write_inspection_log
from src.storage.parquet_reader import get_inspection_summary


def main():
    parser = argparse.ArgumentParser(description="Run autonomous underwater ROV hull inspection simulation.")
    parser.add_argument("--steps", type=int, default=150, help="Max simulation steps (default: 150)")
    parser.add_argument("--turbidity", type=float, default=0.20, help="Optical turbidity level [0.0 - 0.8]")
    parser.add_argument("--dataset", choices=["mavecodd", "crack", "sample"], default="mavecodd",
                        help="Dataset source for ground truth defect observations")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTONOMOUS UNDERWATER ROV SHIP HULL INSPECTION SIMULATION")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("random_seed", 42))
    dev = get_device(args.device)

    # Load test/val samples for observation pool
    manifest_file = ROOT / config["paths"]["processed_dir"] / f"split_manifest_{args.dataset}.json"
    sample_pool = []
    if manifest_file.exists():
        splits = load_split_manifest(manifest_file)
        # Held-out TEST scenes only (val was used for threshold calibration)
        sample_pool = splits.get("test", [])

    # Load trained models
    checkpoints_dir = ROOT / config["paths"]["model_dir"]
    baseline_path = checkpoints_dir / f"baseline_{args.dataset}.joblib"
    unet_path = checkpoints_dir / f"unet_{args.dataset}.pth"

    predictor = None
    if baseline_path.exists() or unet_path.exists():
        print(f"Loading trained models from {checkpoints_dir}...")
        predictor = HullInspectionPredictor(
            baseline_path=baseline_path if baseline_path.exists() else None,
            unet_path=unet_path if unet_path.exists() else None,
            device=dev
        )
    else:
        print("Warning: Trained checkpoints not found. Running simulation without live model predictions.")

    # 1. Initialize Hull Geometry and ground-truth defects
    hull_cfg = config["simulation"]["hull"]
    hull = ShipHullSurface(
        length_m=hull_cfg["length_m"],
        height_m=hull_cfg["height_m"],
        resolution_m=hull_cfg["grid_resolution_m"],
        zone_count_x=hull_cfg["zone_count_x"],
        zone_count_y=hull_cfg["zone_count_y"]
    )

    # Register known defect regions on hull
    defect_coords = [
        (8.5, 3.2, 0.9, "corrosion"),
        (18.0, 7.5, 1.2, "corrosion"),
        (28.5, 4.0, 0.8, "corrosion"),
        (38.0, 6.8, 1.1, "corrosion"),
        (44.5, 2.5, 0.7, "corrosion")
    ]
    for idx, (dx, dy, dr, dtype) in enumerate(defect_coords):
        hull.add_defect(x_m=dx, y_m=dy, radius_m=dr, defect_type=dtype)

    print(f"Hull Surface initialized: {hull.length_m}m x {hull.height_m}m with {len(hull.defects)} registered defects.")

    # 2. Initialize ROV and Camera
    rov_cfg = config["simulation"]["rov"]
    rov = UnderwaterROV(
        x=1.0, y=1.0, z=rov_cfg["inspection_distance_m"],
        speed_mps=rov_cfg["speed_mps"],
        position_noise_std=rov_cfg["position_noise_std"],
        heading_noise_std=rov_cfg["heading_noise_std"],
        random_seed=config.get("random_seed", 42)
    )

    camera = SimulatedUnderwaterCamera(
        fov_deg=rov_cfg["camera_fov_deg"],
        turbidity=args.turbidity,
        red_attenuation=config["simulation"]["environment"]["red_absorption_factor"],
        contrast_factor=1.0 - config["simulation"]["environment"]["contrast_reduction"]
    )

    # 3. Initialize Simulator
    simulator = HullInspectionSimulator(
        hull=hull,
        rov=rov,
        camera=camera,
        predictor=predictor,
        sample_pool=sample_pool
    )

    print(f"Beginning Autonomous Serpentine Inspection (ID: {simulator.inspection_id})...")
    print(f"Planned Waypoints: {len(simulator.waypoints)} | Max Steps: {args.steps} | Turbidity: {args.turbidity:.2f}")

    # 4. Run Mission
    sim_result = simulator.run_full_inspection(max_steps=args.steps)
    print(f"Inspection Run Completed in {sim_result['total_steps']} steps.")
    print(f"Hull Surface Inspected: {sim_result['final_coverage_pct']:.1f}%")
    print(f"Defect Frames Flagged: {sim_result['defects_detected']}")
    sc = sim_result["detection_scores"]
    if predictor is not None and sc["frames"]:
        fmt = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
        print("\n--- Frame-level detection vs ground truth ---")
        print(f"Frames: {sc['frames']} | TP {sc['TP']}  FP {sc['FP']}  FN {sc['FN']}  TN {sc['TN']}")
        print(f"Accuracy {fmt(sc['accuracy'])} | Precision {fmt(sc['precision'])} | "
              f"Recall {fmt(sc['recall'])} | Specificity {fmt(sc['specificity'])}")

    # 5. Save to PyArrow Parquet
    logs_dir = ROOT / config["paths"]["logs_dir"]
    parquet_path = logs_dir / f"{simulator.inspection_id}.parquet"
    write_inspection_log(simulator.inspection_logs, parquet_path)
    print(f"Inspection Telemetry & Defect Log saved to Parquet: {parquet_path}")

    # 6. Generate Summary Analysis
    summary = get_inspection_summary(parquet_path)
    print("\n--- Analytical Inspection Summary ---")
    print(f"Inspection ID: {summary['inspection_id']}")
    print(f"Total Steps Logged: {summary['total_steps']}")
    print(f"Final Coverage: {summary['final_coverage_pct']}%")
    print(f"Suspected Defects Flagged: {summary['total_suspected_defects']}")
    print(f"Mean Detection Confidence: {summary['mean_confidence']}")
    print(f"Defect Breakdown by Zone: {summary['defects_by_zone']}")
    print(f"Battery Remaining: {summary['battery_remaining_pct']}%")

    # Check for missed regions
    missed = hull.get_missed_regions()
    if missed:
        print(f"\nIdentified {len(missed)} missed hull sectors:")
        for m in missed[:3]:
            print(f"  - Zone {m['zone_id']} at ({m['x_center_m']}m, {m['y_center_m']}m)")
    else:
        print("\nComplete inspection achieved! No significant uninspected hull sectors.")

    print("=" * 70)


if __name__ == "__main__":
    main()
