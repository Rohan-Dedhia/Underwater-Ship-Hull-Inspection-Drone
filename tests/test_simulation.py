"""Unit tests for ROV kinematics, hull geometry, and camera projection."""

import pytest
import numpy as np

from src.simulation.hull import ShipHullSurface
from src.simulation.rov import UnderwaterROV
from src.simulation.camera import SimulatedUnderwaterCamera
from src.simulation.inspection_path import SerpentinePathPlanner


def test_ship_hull_surface_and_coverage():
    """Test hull dimensioning, zone queries, and coverage marking."""
    hull = ShipHullSurface(length_m=40.0, height_m=10.0, resolution_m=1.0)

    assert hull.get_coverage_percentage() == 0.0

    # Mark a 5m x 5m region
    covered = hull.mark_coverage(x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0)
    assert covered > 0
    assert hull.get_coverage_percentage() > 0.0

    # Zone query
    zone = hull.get_zone_at(2.0, 2.0)
    assert "Bow" in zone or "Sec" in zone


def test_underwater_rov_kinematics():
    """Test ROV position update and noisy telemetry."""
    rov = UnderwaterROV(x=0.0, y=0.0, z=1.5, speed_mps=1.0, random_seed=42)

    # Move towards (5.0, 0.0, 1.5) with dt = 1.0s -> moves 1.0m
    pose = rov.update_motion(target_x=5.0, target_y=0.0, target_z=1.5, dt_s=1.0)

    assert 0.99 <= rov.x <= 1.01
    assert rov.total_distance_m >= 0.99
    assert "measured_x" in pose
    # Measured coordinates should have slight Gaussian offset from true coordinates
    assert abs(pose["measured_x"] - pose["true_x"]) < 0.25


def test_camera_footprint_calculation():
    """Test camera footprint scaling with standoff distance."""
    cam = SimulatedUnderwaterCamera(fov_deg=90.0, aspect_ratio=1.0)
    # At 90 deg FOV, half angle is 45 deg, tan(45)=1.0 -> width = 2 * z * 1.0 = 2 * z
    w, h = cam.calculate_footprint(standoff_z_m=2.0)
    assert abs(w - 4.0) < 0.01
    assert abs(h - 4.0) < 0.01


def test_serpentine_path_coverage():
    """Test serpentine waypoints span the hull length and height."""
    planner = SerpentinePathPlanner(hull_length_m=20.0, hull_height_m=6.0, step_x_m=2.0, step_y_m=1.5)
    wps = planner.generate_serpentine_waypoints()

    assert len(wps) > 10
    xs = [p[0] for p in wps]
    ys = [p[1] for p in wps]

    assert min(xs) >= 0.5
    assert max(xs) <= 20.0
    assert min(ys) >= 0.5
    assert max(ys) <= 6.0


def test_lidar_sensor_scan():
    """Test multi-beam LiDAR scan ranges and obstacle detection."""
    from src.simulation.sensors import LidarSensor
    lidar = LidarSensor(num_beams=19, fov_deg=90.0)
    res = lidar.scan(drone_x=5.0, drone_y=5.0, standoff_z=1.5)

    assert res["num_beams"] == 19
    assert len(res["ranges_m"]) == 19
    assert 1.40 <= res["min_distance_m"] <= 1.60
    assert len(res["point_cloud"]) == 19


def test_pressure_depth_sensor():
    """Test hydrostatic pressure calculation with depth."""
    from src.simulation.sensors import PressureDepthSensor
    sensor = PressureDepthSensor(waterline_y_m=10.0)

    # At keel (y=0m), depth = 10m -> P approx 1.013 + 1.005 bar approx 2.01 bar
    res = sensor.read(drone_y_m=0.0)
    assert 9.9 <= res["submerged_depth_m"] <= 10.1
    assert 1.95 <= res["hydrostatic_pressure_bar"] <= 2.10

    # At surface (y=10m), depth = 0m -> P approx 1.013 bar
    res_surf = sensor.read(drone_y_m=10.0)
    assert res_surf["submerged_depth_m"] == 0.0
    assert 0.98 <= res_surf["hydrostatic_pressure_bar"] <= 1.05


def test_collision_detection_system():
    """Test proximity warning thresholds and auto-brake action."""
    from src.simulation.sensors import CollisionDetectionSystem
    cds = CollisionDetectionSystem(safe_standoff_m=1.2, warning_standoff_m=0.8, critical_standoff_m=0.5)

    safe = cds.evaluate(current_standoff_m=1.5)
    assert safe["alert_level"] == "SAFE"
    assert safe["auto_brake_active"] is False

    warn = cds.evaluate(current_standoff_m=0.75)
    assert warn["alert_level"] == "WARNING"

    crit = cds.evaluate(current_standoff_m=0.40)
    assert crit["alert_level"] == "CRITICAL_COLLISION"
    assert crit["auto_brake_active"] is True


def test_hud_rendering():
    """Test HUD cockpit stamping on camera frame."""
    cam = SimulatedUnderwaterCamera(resolution=(320, 240))
    dummy_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    hud = cam.render_hud(dummy_frame)
    assert hud.shape == (240, 320, 3)
    # HUD should have drawn graphics (reticle, text) modifying frame values
    assert not np.array_equal(hud, dummy_frame)


def test_synthetic_camera_frames_carry_ground_truth():
    """Without real photos, frames over a registered defect contain painted corrosion and GT says so."""
    from src.simulation.hull import ShipHullSurface
    cam = SimulatedUnderwaterCamera(resolution=(320, 240), frame_samples=[])
    hull = ShipHullSurface(length_m=20.0, height_m=10.0)
    hull.add_defect(10.0, 5.0, 1.0, "corrosion")
    over = cam.capture_observation(10.0, 5.0, 1.5, hull.defects)
    away = cam.capture_observation(2.0, 2.0, 1.5, hull.defects)
    assert over["ground_truth_defect"] is True and over["ground_truth_mask"].mean() > 0.05
    assert away["ground_truth_defect"] is False and away["ground_truth_mask"].sum() == 0


def test_simulator_scores_frames_against_ground_truth():
    from src.simulation.hull import ShipHullSurface
    from src.simulation.simulator import HullInspectionSimulator
    hull = ShipHullSurface(length_m=20.0, height_m=6.0)
    hull.add_defect(3.0, 1.2, 1.0, "corrosion")
    sim = HullInspectionSimulator(hull=hull, camera=SimulatedUnderwaterCamera(resolution=(320, 240), frame_samples=[]))
    for _ in range(10):
        sim.step(dt_s=0.5)
    # Without a predictor nothing is flagged: every GT-defect frame counts as a miss
    sc = sim.get_detection_scores()
    assert sc["frames"] == 10
    assert sc["TP"] == 0 and sc["FP"] == 0
    assert sc["FN"] == sum(r["ground_truth_defect"] for r in sim.inspection_logs)
