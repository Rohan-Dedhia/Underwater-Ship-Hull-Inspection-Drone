"""Streamlit Page 3: Autonomous Underwater ROV Simulation, Live HUD Camera Feed, and Multi-Sensor Telemetry."""

import streamlit as st
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Drone Simulation | Hull AI", layout="wide")

from src.utils.reproducibility import get_device
from src.simulation.hull import ShipHullSurface
from src.simulation.rov import UnderwaterROV
from src.simulation.camera import SimulatedUnderwaterCamera
from src.simulation.simulator import HullInspectionSimulator
from src.models.predict import HullInspectionPredictor
from src.storage.parquet_writer import write_inspection_log
from app.ui import inject_css, callout, badge, page_header

inject_css()
page_header("🤖 Drone simulation",
            "The ROV flies a lawnmower path over a 50 m × 10 m hull with 5 registered corrosion areas. "
            "Every camera frame is analysed by the U-Net and scored against its ground truth.",
            chips=["Frames from held-out test photos", "Ground truth per frame", "Live scorecard"])

with st.expander("ℹ️ What am I looking at? (read once)"):
    callout("When the camera footprint is over one of the red corrosion circles on the map, the camera shows a "
            "corroded photo; elsewhere it shows clean hull plate. All frames are cut from <b>held-out test photos</b> "
            "the model never trained on, restricted to windows that show hull plate (no sky or deck railings), and "
            "every frame has a ground-truth mask.<br/><b>Keep in mind:</b> this is an easy setting — plain paint "
            "versus heavy corrosion. The harder, more representative numbers (including false alarms on weathered "
            "paint and deck structures) are on the <b>Model evaluation</b> page.", "info")

U_PATH = ROOT / "models" / "checkpoints" / "unet_mavecodd.pth"


# Load the U-Net once for real-time inference (cache refreshes when the checkpoint changes)
@st.cache_resource
def get_cached_predictor(mtime: float):
    if U_PATH.exists():
        return HullInspectionPredictor(unet_path=U_PATH, device=get_device("auto"))
    return None


predictor = get_cached_predictor(U_PATH.stat().st_mtime if U_PATH.exists() else 0.0)
if predictor is None:
    callout("No trained U-Net found — frames are shown without AI analysis. Run "
            "<b>python scripts/train_model.py --dataset mavecodd</b>.", "warn")

DEFECTS = [(8.5, 3.2, 1.0), (18.0, 7.5, 1.2), (28.5, 4.0, 0.9), (38.0, 6.8, 1.1), (44.5, 2.5, 0.8)]


@st.cache_resource
def get_frame_bank():
    """Held-out camera frames are expensive to cut, so build them once per server."""
    return SimulatedUnderwaterCamera().get_frame_bank()


def new_simulator(standoff: float, speed: float, turbidity: float, red_att: float) -> HullInspectionSimulator:
    hull = ShipHullSurface(length_m=50.0, height_m=10.0, resolution_m=0.5)
    for x, y, r in DEFECTS:
        hull.add_defect(x, y, r, "corrosion")
    rov = UnderwaterROV(x=1.0, y=1.0, z=standoff, speed_mps=speed, random_seed=42)
    cam = SimulatedUnderwaterCamera(fov_deg=75.0, turbidity=turbidity, red_attenuation=red_att, resolution=(640, 480))
    cam._frame_bank = get_frame_bank()
    return HullInspectionSimulator(hull=hull, rov=rov, camera=cam, predictor=predictor)


# Simulation state management
if "sim" not in st.session_state or st.session_state.get("sim_predictor") is not predictor:
    st.session_state["sim"] = new_simulator(1.5, 0.8, 0.20, 0.50)
    st.session_state["sim_predictor"] = predictor
    st.session_state["last_step_res"] = None

sim: HullInspectionSimulator = st.session_state["sim"]
bank = sim.camera.get_frame_bank()
if not (bank["sound"] and bank["defect"]):
    callout("Held-out MaVeCoDD test photos not found, so the camera shows <b>synthetic</b> plates with painted "
            "corrosion. Run <b>python scripts/prepare_dataset.py</b> with the dataset present for real photos.", "warn")

# Sidebar: environment and vehicle settings
st.sidebar.markdown("#### 🕹️ ROV & water settings")
sim.rov.speed_mps = st.sidebar.slider("Cruise speed (m/s)", 0.2, 2.5, float(sim.rov.speed_mps), 0.1)
sim.camera.turbidity = st.sidebar.slider("Water turbidity", 0.0, 0.7, float(sim.camera.turbidity), 0.05,
                                         help="0 = clear water. Higher values add haze, blur and marine snow.")
sim.camera.red_attenuation = st.sidebar.slider("Red light absorption", 0.0, 0.9, float(sim.camera.red_attenuation), 0.05)
standoff = st.sidebar.slider(
    "Distance to hull (m)", min_value=0.30, max_value=3.0, value=float(sim.rov.z), step=0.05,
    help="< 0.8 m triggers a WARNING; < 0.5 m triggers a CRITICAL COLLISION alert."
)
sim.rov.z = standoff

# Mission control bar
with st.container(border=True):
    c_b1, c_b2, c_b3, c_live, c_save, c_reset = st.columns([1, 1, 1, 1.2, 1, 1])
    btn_step = c_b1.button("Step", icon=":material/skip_next:", width="stretch")
    btn_step10 = c_b2.button("+10 steps", icon=":material/fast_forward:", width="stretch")
    btn_step50 = c_b3.button("+50 steps", icon=":material/double_arrow:", width="stretch", type="primary")
    live_stream = c_live.toggle("Live video", value=False, help="Plays 15 frames as a short video")
    btn_save = c_save.button("Save log", icon=":material/save:", width="stretch")
    btn_reset = c_reset.button("Reset", icon=":material/restart_alt:", width="stretch")

if btn_reset:
    st.session_state["sim"] = new_simulator(standoff, sim.rov.speed_mps, sim.camera.turbidity, sim.camera.red_attenuation)
    st.session_state["last_step_res"] = None
    st.rerun()

steps_to_run = 50 if btn_step50 else (10 if btn_step10 else (1 if btn_step else 0))
if steps_to_run > 0:
    last_res = None
    with st.spinner(f"Flying {steps_to_run} steps and analysing each camera frame..."):
        for _ in range(steps_to_run):
            last_res = sim.step(dt_s=0.6)
            if sim.is_completed:
                break
    if last_res and "record" in last_res:
        st.session_state["last_step_res"] = last_res

if btn_save and sim.inspection_logs:
    logs_dir = ROOT / "outputs" / "inspection_logs"
    save_path = logs_dir / f"{sim.inspection_id}.parquet"
    write_inspection_log(sim.inspection_logs, save_path)
    st.success(f"Saved {len(sim.inspection_logs)} observation steps to `{save_path.name}` — open it on the "
               "Inspection reports page.")

# Layout containers are created in display order; they are filled after the simulation advances
score_area = st.container()
tab_cam, tab_sensors, tab_log = st.tabs(["📹 Camera & hull map", "📡 Sensors", "📋 Telemetry log"])
with tab_cam:
    col_feed, col_info = st.columns([3, 2])
    with col_feed:
        camera_slot = st.empty()

current_res = st.session_state.get("last_step_res")
if current_res is None and not sim.inspection_logs:
    current_res = sim.step(dt_s=0.1)
    st.session_state["last_step_res"] = current_res

if live_stream:
    for _ in range(15):
        step_res = sim.step(dt_s=0.5)
        if "record" not in step_res:
            break
        current_res = step_res
        st.session_state["last_step_res"] = current_res
        hud_img = current_res["observation"].get("hud_image", current_res["observation"]["image"])
        camera_slot.image(hud_img, caption="🔴 LIVE ROV INSPECTION FEED", width="stretch")
        time.sleep(0.08)

sensor_data = current_res.get("sensor_data", {}) if current_res else {}
dp_data = sensor_data.get("depth_pressure", {})
lidar_data = sensor_data.get("lidar", {})
col_data = sensor_data.get("collision", {})
imu_data = sensor_data.get("imu", {})

# ---------------- Scorecard ----------------
with score_area:
    if sim.is_completed:
        st.success("Inspection path completed.")
    q0, q1, q2, q3, q4, q5 = st.columns(6)
    q0.metric("Coverage", f"{sim.hull.get_coverage_percentage():.1f}%", help="Share of the hull surface the camera has seen")
    if predictor is not None and sim.inspection_logs:
        sc = sim.get_detection_scores()
        pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
        q1.metric("Frames", sc["frames"], help="Camera frames analysed and scored so far")
        q2.metric("Accuracy", pct(sc["accuracy"]), help="Correct decisions: (correct alarms + correct all-clears) / frames")
        q3.metric("Caught", f"{sc['TP']} / {sc['TP'] + sc['FN']}",
                  help="Recall: frames with real corrosion that raised an alarm")
        q4.metric("False alarms", f"{sc['FP']} / {sc['FP'] + sc['TN']}", help="Clean-plate frames that raised an alarm (out of all clean frames)")
        q5.metric("Missed", sc["FN"], help="Frames with real corrosion and no alarm")
        if sc["TP"] + sc["FN"] == 0:
            st.caption("No frame over a registered corrosion area yet. The path scans the bottom rows first and "
                       "reaches the first corrosion area after roughly 110–130 steps — press **+50 steps** a few times.")

# ---------------- Camera + map tab ----------------
with tab_cam:
    with col_feed:
        if current_res and "observation" in current_res:
            obs = current_res["observation"]
            camera_slot.image(obs.get("hud_image", obs["image"]),
                              caption=f"ROV camera · frame #{obs.get('frame_number', 1)} · zone "
                                      f"{current_res['record']['zone_id']}", width="stretch")
    with col_info:
        if current_res and "observation" in current_res and current_res.get("ml_result"):
            obs, ml_res = current_res["observation"], current_res["ml_result"]
            gt_def = bool(obs.get("ground_truth_defect"))
            pred_def = bool(ml_res["defect_detected"])
            truth_txt = "corrosion" if gt_def else "clean plate"
            pred_txt = "alarm" if pred_def else "no alarm"
            if gt_def == pred_def:
                verdict = badge(f"✅ Correct: {pred_txt} on {truth_txt}", "good")
            elif pred_def:
                verdict = badge(f"⚠️ False alarm on {truth_txt}", "fair")
            else:
                verdict = badge(f"❌ Missed {truth_txt}", "poor")
            st.markdown(f"##### This frame\n{verdict}", unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            f1.metric("True corrosion", f"{obs.get('ground_truth_defect_frac', 0) * 100:.1f}%")
            f2.metric("Model flagged", f"{ml_res['defect_area_fraction'] * 100:.1f}%",
                      help=f"Alarm when ≥ {predictor.detection_area_threshold * 100:.0f}% of the frame is flagged")
            gt_mask = obs.get("ground_truth_mask")
            if gt_mask is not None:
                g1, g2 = st.columns(2)
                clean = obs["clean_image"].copy()
                gm = gt_mask > 0
                clean[gm] = (0.45 * clean[gm] + 0.55 * np.array([255, 200, 0])).astype(np.uint8)
                g1.image(clean, caption="Ground truth (yellow)", width="stretch")
                g2.image(HullInspectionPredictor.draw_overlay(obs["image"], ml_res, show_boxes=False),
                         caption="Prediction (magenta)", width="stretch")
            st.markdown(f"""<div class="hud-status">
                CONFIDENCE {ml_res['confidence_score']*100:.0f}% &bull; LATENCY {ml_res['inference_time_ms']:.0f} ms
            </div>""", unsafe_allow_html=True)

    # 2D Hull Map
    fig_map = go.Figure()
    fig_map.add_shape(type="rect", x0=0, y0=0, x1=sim.hull.length_m, y1=sim.hull.height_m,
                      line=dict(color="#1e3d59", width=3), fillcolor="#f0f4f8")
    for z_id, z in sim.hull.zones.items():
        fig_map.add_shape(type="rect", x0=z["x_min"], y0=z["y_min"], x1=z["x_max"], y1=z["y_max"],
                          line=dict(color="#b0bec5", width=1, dash="dot"))
        fig_map.add_annotation(x=(z["x_min"] + z["x_max"]) / 2, y=z["y_max"] - 0.4,
                               text=z_id.split("-")[0], showarrow=False, font=dict(size=9, color="#546e7a"))
    cov_y, cov_x = np.where(sim.hull.coverage_grid == 1)
    if len(cov_x) > 0:
        fig_map.add_trace(go.Scatter(x=cov_x * sim.hull.resolution_m, y=cov_y * sim.hull.resolution_m,
                                     mode="markers", name="Inspected cells",
                                     marker=dict(size=6, color="#48dbfb", symbol="square", opacity=0.35)))
    wps = sim.waypoints
    fig_map.add_trace(go.Scatter(x=[p[0] for p in wps], y=[p[1] for p in wps], mode="lines",
                                 line=dict(color="#90a4ae", width=1, dash="dash"), name="Planned path"))
    for d in sim.hull.defects:
        fig_map.add_shape(type="circle", x0=d["x_m"] - d["radius_m"], y0=d["y_m"] - d["radius_m"],
                          x1=d["x_m"] + d["radius_m"], y1=d["y_m"] + d["radius_m"],
                          line=dict(color="#e74c3c", width=2), fillcolor="rgba(231, 76, 60, 0.25)")
        fig_map.add_annotation(x=d["x_m"], y=d["y_m"], text=d["defect_id"], showarrow=False,
                               font=dict(size=10, color="#8e1b10"))
    for label, cond, color, symbol in [
        ("Correct alarm", lambda r: r["defect_detected"] and r["ground_truth_defect"], "#1e9e4a", "x"),
        ("False alarm", lambda r: r["defect_detected"] and not r["ground_truth_defect"], "#e67e22", "x"),
        ("Missed defect", lambda r: (not r["defect_detected"]) and r["ground_truth_defect"], "#2155d6", "circle-open"),
    ]:
        pts = [r for r in sim.inspection_logs if cond(r)]
        if pts:
            fig_map.add_trace(go.Scatter(x=[r["drone_x"] for r in pts], y=[r["drone_y"] for r in pts],
                                         mode="markers", name=label,
                                         marker=dict(size=10, color=color, symbol=symbol, line=dict(width=2, color=color))))
    fig_map.add_trace(go.Scatter(x=[sim.rov.x], y=[sim.rov.y], mode="markers+text", name="ROV",
                                 marker=dict(size=16, color="#0984e3", symbol="diamond"),
                                 text=["ROV"], textposition="top center", textfont=dict(color="#0b3a66", size=13)))
    fw, fh = sim.camera.calculate_footprint(sim.rov.z)
    fig_map.add_shape(type="rect", x0=sim.rov.x - fw / 2, y0=sim.rov.y - fh / 2,
                      x1=sim.rov.x + fw / 2, y1=sim.rov.y + fh / 2,
                      line=dict(color="#0984e3", width=2), fillcolor="rgba(9, 132, 227, 0.20)")
    fig_map.update_layout(
        title="Hull map · red circles = corrosion areas · blue box = camera footprint",
        xaxis=dict(range=[-1, sim.hull.length_m + 1], title="Bow → stern (m)"),
        yaxis=dict(range=[-1, sim.hull.height_m + 1], title="Keel → waterline (m)", scaleanchor="x", scaleratio=1),
        height=400, margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    )
    st.plotly_chart(fig_map, width="stretch")

# ---------------- Sensors tab ----------------
with tab_sensors:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Submerged depth", f"{dp_data.get('submerged_depth_m', 9.0):.2f} m")
    k2.metric("Hydrostatic pressure", f"{dp_data.get('hydrostatic_pressure_bar', 1.91):.2f} bar")
    k3.metric("LiDAR standoff", f"{lidar_data.get('min_distance_m', sim.rov.z):.2f} m")
    k4.metric("Battery", f"{sim.rov.battery_percent:.1f}%")
    k5.metric("Distance travelled", f"{sim.rov.total_distance_m:.1f} m")

    col_radar, col_diag = st.columns([1, 1])
    with col_radar:
        if lidar_data and "angles_deg" in lidar_data:
            angles, ranges = lidar_data["angles_deg"], lidar_data["ranges_m"]
            beam_colors = ["#e74c3c" if r < 0.50 else "#f39c12" if r < 0.80 else "#f1c40f" if r < 1.20 else "#2ecc71"
                           for r in ranges]
            fig_lidar = go.Figure()
            fig_lidar.add_trace(go.Barpolar(r=ranges, theta=[a + 90 for a in angles], width=[4.5] * len(angles),
                                            marker=dict(color=beam_colors, line=dict(color="#111", width=1)),
                                            name="LiDAR beams", hoverinfo="r+theta"))
            fig_lidar.add_trace(go.Scatterpolar(r=[0.50] * 36, theta=np.linspace(0, 360, 36), mode="lines",
                                                line=dict(color="#e74c3c", width=1.5, dash="dot"),
                                                name="Critical 0.5 m perimeter"))
            fig_lidar.update_layout(
                title="LiDAR scan (colour = distance band)",
                polar=dict(radialaxis=dict(range=[0, max(3.5, max(ranges) + 0.5)], visible=True, showticklabels=True),
                           angularaxis=dict(direction="clockwise", rotation=90)),
                height=330, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig_lidar, width="stretch")

        alert_lvl = col_data.get("alert_level", "SAFE")
        alert_col = col_data.get("alert_color", "#2ecc71")
        action_text = col_data.get("evasive_action", "NOMINAL_SURVEY")
        # Darker shades of the alert colours so the status word stays readable on the light banner
        alert_text = {"#e74c3c": "#a8251a", "#f39c12": "#8a5300", "#f1c40f": "#7a6200",
                      "#2ecc71": "#1b7a43"}.get(alert_col, "#1f2933")
        st.markdown(f"""
        <div style="background-color: #f7f9fa; border: 2px solid {alert_col}; border-left: 8px solid {alert_col};
                    border-radius: 8px; padding: 12px; margin-top: 5px; color: #1f2933;">
            <span style="font-weight: 700; color: #1f2933; font-size: 1.05rem;">
                🛡️ Collision avoidance: <span style="color: {alert_text};">{alert_lvl}</span>
            </span><br/>
            <span style="font-size: 0.85rem; color: #1f2933;">
                <b>Hull clearance:</b> {col_data.get('clearance_margin_m', 0.8):.2f} m &bull;
                <b>Action:</b> {action_text} &bull;
                <b>Auto-brake:</b> {'ACTIVE' if col_data.get('auto_brake_active') else 'STANDBY'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_diag:
        env = sensor_data.get("environment", {})
        with st.container(border=True):
            st.markdown("##### 🌊 Depth & pressure")
            st.markdown(f"Depth **{dp_data.get('submerged_depth_m', 9.0):.2f} m** below the waterline · "
                        f"**{dp_data.get('hydrostatic_pressure_bar', 1.91):.3f} bar** "
                        f"({dp_data.get('hydrostatic_pressure_psi', 27.7):.1f} PSI) · "
                        f"water {dp_data.get('water_temp_c', 14.5):.1f} °C")
        with st.container(border=True):
            st.markdown("##### 🧭 IMU attitude")
            st.markdown(f"Roll **{imu_data.get('roll_deg', 0.0):+.2f}°** · pitch **{imu_data.get('pitch_deg', 0.0):+.2f}°** "
                        f"· heading **{imu_data.get('yaw_deg', 0.0):.1f}°** · gyro "
                        f"{imu_data.get('gyro_x_dps', 0.0):+.2f} / {imu_data.get('gyro_y_dps', 0.0):+.2f} °/s · "
                        f"a<sub>z</sub> {imu_data.get('accel_z_mps2', 9.81):.2f} m/s²", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("##### 🧪 Water quality")
            st.markdown(f"Turbidity **{env.get('turbidity_ntu', 18.5)} NTU** · salinity "
                        f"**{env.get('salinity_psu', 35.2)} PSU**")

# ---------------- Telemetry tab ----------------
with tab_log:
    if sim.inspection_logs:
        df_logs = pd.DataFrame(sim.inspection_logs)
        df_logs["outcome"] = np.select(
            [df_logs.defect_detected & df_logs.ground_truth_defect, df_logs.defect_detected & ~df_logs.ground_truth_defect,
             ~df_logs.defect_detected & df_logs.ground_truth_defect],
            ["✅ correct alarm", "⚠️ false alarm", "❌ missed"], default="✅ correct all-clear")
        st.dataframe(
            df_logs[["step_id", "drone_x", "drone_y", "zone_id", "ground_truth_defect", "defect_detected", "outcome",
                     "confidence_score", "coverage_pct"]].iloc[::-1].head(50),
            width="stretch", hide_index=True,
            column_config={"drone_x": st.column_config.NumberColumn("x (m)", format="%.1f"),
                           "drone_y": st.column_config.NumberColumn("y (m)", format="%.1f"),
                           "confidence_score": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1),
                           "coverage_pct": st.column_config.NumberColumn("coverage %", format="%.1f")})
        st.caption("Newest 50 frames first. Use **Save log** to store the whole mission as Parquet.")
    else:
        st.info("No frames logged yet.")
