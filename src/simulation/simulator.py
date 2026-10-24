"""Integrated ROV Hull Inspection Simulator with real-time ML inference."""

from typing import Dict, Any, List, Optional
import time
import uuid
from datetime import datetime
import numpy as np

from src.simulation.hull import ShipHullSurface
from src.simulation.rov import UnderwaterROV
from src.simulation.inspection_path import SerpentinePathPlanner
from src.simulation.camera import SimulatedUnderwaterCamera
from src.simulation.sensors import ROVSensorSuite
from src.models.predict import HullInspectionPredictor


class HullInspectionSimulator:
    """End-to-end autonomous hull inspection simulator integrating robotics and machine learning."""

    def __init__(
        self,
        hull: Optional[ShipHullSurface] = None,
        rov: Optional[UnderwaterROV] = None,
        camera: Optional[SimulatedUnderwaterCamera] = None,
        predictor: Optional[HullInspectionPredictor] = None,
        inspection_id: Optional[str] = None,
        sample_pool: Optional[List[Dict[str, Any]]] = None
    ):
        self.hull = hull or ShipHullSurface(length_m=50.0, height_m=10.0)
        self.rov = rov or UnderwaterROV(x=1.0, y=1.0, z=1.5, speed_mps=0.8)
        self.camera = camera or SimulatedUnderwaterCamera()
        self.predictor = predictor
        self.sample_pool = sample_pool or []

        self.sensor_suite = ROVSensorSuite(waterline_y_m=self.hull.height_m, safe_standoff_m=1.2)
        self.inspection_id = inspection_id or f"INSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.planner = SerpentinePathPlanner(
            hull_length_m=self.hull.length_m,
            hull_height_m=self.hull.height_m,
            standoff_dist_m=self.rov.z,
            step_x_m=2.5,
            step_y_m=1.2
        )
        self.waypoints = self.planner.generate_serpentine_waypoints()
        self.current_waypoint_idx = 0
        self.inspection_logs: List[Dict[str, Any]] = []
        self.step_counter = 0
        self.is_completed = False

    def step(self, dt_s: float = 0.5) -> Dict[str, Any]:
        """Execute one simulation timestep."""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.is_completed = True
            return {"status": "completed", "coverage_pct": self.hull.get_coverage_percentage()}

        target = self.waypoints[self.current_waypoint_idx]
        pose = self.rov.update_motion(target[0], target[1], target[2], dt_s)

        # Check if current waypoint reached
        dist_to_target = np.sqrt(
            (self.rov.x - target[0])**2 +
            (self.rov.y - target[1])**2 +
            (self.rov.z - target[2])**2
        )
        if dist_to_target < 0.3:
            self.current_waypoint_idx += 1

        # Camera observation
        obs = self.camera.capture_observation(
            self.rov.x,
            self.rov.y,
            self.rov.z,
            self.hull.defects,
            self.sample_pool
        )

        # Mark hull coverage
        x_min, x_max, y_min, y_max = obs["footprint_bounds"]
        self.hull.mark_coverage(x_min, x_max, y_min, y_max)

        # Run ML model inference if predictor available
        ml_result = None
        pred_class = "unprocessed"
        confidence = 0.0
        status = "pending"
        bbox_str = None
        has_defect = False

        if self.predictor is not None:
            ml_result = self.predictor.predict(obs["image"], run_baseline=False)
            pred_class = ml_result["predicted_defect_class"]
            confidence = ml_result["confidence_score"]
            status = ml_result["status"]
            has_defect = ml_result["defect_detected"]
            if ml_result["bounding_boxes"]:
                b = ml_result["bounding_boxes"][0]["bbox_normalized"]
                bbox_str = f"[{b[0]},{b[1]},{b[2]},{b[3]}]"

        self.step_counter += 1
        zone_id = self.hull.get_zone_at(self.rov.x, self.rov.y)

        # Sample all sensors (LiDAR, pressure, depth, collision, IMU)
        sensor_data = self.sensor_suite.sample_all(
            drone_x=self.rov.x,
            drone_y=self.rov.y,
            standoff_z=self.rov.z,
            heading_deg=pose["true_heading_deg"],
            speed_mps=self.rov.speed_mps,
            approach_velocity=self.rov.vz
        )

        # Render tactical HUD overlay onto camera frame
        obs["hud_image"] = self.camera.render_hud(
            obs["image"],
            sensor_data=sensor_data,
            ml_result=ml_result,
            zone_id=zone_id
        )

        # Structured telemetry log record
        record = {
            "inspection_id": self.inspection_id,
            "step_id": self.step_counter,
            "timestamp": datetime.now().isoformat(),
            "drone_x": pose["true_x"],
            "drone_y": pose["true_y"],
            "drone_z": pose["true_z"],
            "drone_heading_deg": pose["true_heading_deg"],
            "measured_x": pose["measured_x"],
            "measured_y": pose["measured_y"],
            "measured_z": pose["measured_z"],
            "measured_heading_deg": pose["measured_heading_deg"],
            "zone_id": zone_id,
            "camera_image_id": obs["camera_image_id"],
            "dataset_source": obs["dataset_source"],
            "ground_truth_defect": obs["ground_truth_defect"],
            "predicted_defect_class": pred_class,
            "confidence_score": round(confidence, 4),
            "defect_detected": has_defect,
            "defect_bbox": bbox_str,
            "simulated_visibility": round(1.0 - obs["simulated_turbidity"], 2),
            "simulated_turbidity": obs["simulated_turbidity"],
            "inspection_status": status,
            "coverage_pct": self.hull.get_coverage_percentage(),
            "battery_pct": pose["battery_percent"],
            "lidar_min_dist_m": sensor_data["lidar"]["min_distance_m"],
            "submerged_depth_m": sensor_data["depth_pressure"]["submerged_depth_m"],
            "pressure_bar": sensor_data["depth_pressure"]["hydrostatic_pressure_bar"],
            "collision_status": sensor_data["collision"]["alert_level"]
        }

        self.inspection_logs.append(record)

        return {
            "record": record,
            "observation": obs,
            "ml_result": ml_result,
            "sensor_data": sensor_data,
            "waypoint_progress": f"{self.current_waypoint_idx}/{len(self.waypoints)}",
            "coverage_pct": self.hull.get_coverage_percentage(),
            "status": "in_progress" if not self.is_completed else "completed"
        }

    def get_detection_scores(self) -> Dict[str, Any]:
        """Score every logged frame: model decision vs the frame's ground truth.

        A frame's ground truth is 'defect' when >= 5% of its pixels are labelled corrosion.
        """
        tp = sum(1 for r in self.inspection_logs if r["defect_detected"] and r["ground_truth_defect"])
        fp = sum(1 for r in self.inspection_logs if r["defect_detected"] and not r["ground_truth_defect"])
        fn = sum(1 for r in self.inspection_logs if not r["defect_detected"] and r["ground_truth_defect"])
        tn = sum(1 for r in self.inspection_logs if not r["defect_detected"] and not r["ground_truth_defect"])
        n = tp + fp + fn + tn
        return {
            "frames": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / n if n else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "specificity": tn / (tn + fp) if (tn + fp) else None,
        }

    def run_full_inspection(self, max_steps: int = 400) -> Dict[str, Any]:
        """Run inspection to completion or step limit."""
        while not self.is_completed and self.step_counter < max_steps:
            self.step(dt_s=0.5)

        return {
            "inspection_id": self.inspection_id,
            "total_steps": self.step_counter,
            "final_coverage_pct": self.hull.get_coverage_percentage(),
            "total_records": len(self.inspection_logs),
            "defects_detected": sum(1 for r in self.inspection_logs if r["defect_detected"]),
            "detection_scores": self.get_detection_scores(),
            "missed_regions": self.hull.get_missed_regions()
        }
