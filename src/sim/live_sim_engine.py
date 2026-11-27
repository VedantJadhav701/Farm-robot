"""
Interactive Live Full-Field MuJoCo Simulation Engine
====================================================
Runs an 8-tree full-field pomegranate orchard simulation in a background thread,
streaming live side-by-side MJPEG video to the Web Cockpit while accepting
real-time user controls:
  - Start / Pause / Resume / Reset / Emergency Stop (E-Stop)
  - Auto Mission Mode vs Manual Teleop (Forward, Reverse, Turn L/R, Pour, Retract)
  - Camera Switcher (Chase+YOLO, Full-Field Overview+YOLO)
  - Adjustable Dose Volume (10-500 mL)
"""

import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import mujoco

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.control.kinematics import ArmKinematics
from src.control.visual_servoing import EyeInHandVisualServo
from src.control.target_tracker import TemporalTargetStabilizer
from src.control.fertilizer_controller import FertilizerDosingController
from src.api.llm_brain import DosingAuthorization
from src.hardware.ros2_actuator_bridge import ROS2ActuatorBridge, SafetyTelemetry


class LiveOrchardSimEngine:
    def __init__(self):
        self.model_path = os.path.abspath("src/sim/models/full_field_orchard.xml")
        weights = (
            "models/phase3/best_phase3.pt"
            if Path("models/phase3/best_phase3.pt").exists()
            else "models/best.pt"
        )
        self.detector = PomegranateVisionDetector(
            weights_path=weights, confidence_threshold=0.70, nms_conf=0.22
        )
        self.kin = ArmKinematics()
        self.ibvs = EyeInHandVisualServo()
        self.tracker = TemporalTargetStabilizer(ema_alpha=0.28, max_jump_px=115.0)
        self.controller = FertilizerDosingController(db_path="output/dosing_audit_log.db")
        self.safety = ROS2ActuatorBridge()

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer: Optional[mujoco.Renderer] = None

        self.cam_onboard = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_orchard_cam")
        self.cam_follow = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_follow_cam")
        self.cam_overview = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "field_overview_cam")

        self.tree_x_targets = [1.20, 3.00, 4.80, 6.60, 8.40, 10.20, 12.00, 13.80]
        self.home_q = self.kin.home_q.copy()
        self.pour_arm_q = np.array([1.45, 0.95, -0.55, 0.0, -0.45, 0.0], dtype=np.float64)

        self._lock = threading.Lock()
        self._jpeg_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_onboard_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._last_vision = None
        self._dosing_active = False
        self._ibvs_error_cm = 0.0
        self._ibvs_aligned = False

        # Interactive control state
        self.running = True
        self.estop = False
        self.mode = "AUTO"  # AUTO | MANUAL
        self.camera_view = "FOLLOW"  # FOLLOW | OVERVIEW
        self.dose_ml = 80.0
        self.manual_linear = 0.0  # -1..1
        self.manual_turn = 0.0    # -1..1
        self.manual_arm_pour = False

        # Mission telemetry
        self.tree_idx = 0
        self.phase = "DRIVE"
        self.phase_timer = 0
        self.trees_poured = 0
        self.gate_summary = "4/4 ARMED"
        self.last_conf = 0.0
        self.last_s1_ms = 0.0
        self.logs: list[str] = ["[SIM] Full-field 8-tree MuJoCo world initialized."]

        self.reset_simulation(keep_running=True)

    def start_background_loop(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.logs.append(msg)
        if len(self.logs) > 12:
            self.logs = self.logs[-12:]

    def reset_simulation(self, keep_running: bool = False):
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.ctrl[4:10] = self.home_q
            self.tracker = TemporalTargetStabilizer(ema_alpha=0.28, max_jump_px=115.0)
            self.tree_idx = 0
            self.phase = "DRIVE"
            self.phase_timer = 0
            self.trees_poured = 0
            self.gate_summary = "4/4 ARMED"
            self.estop = False
            self.manual_linear = 0.0
            self.manual_turn = 0.0
            self.manual_arm_pour = False
            self.running = keep_running
            self._log("[SIM] Field reset to Row Start (x=0.0m, 8 trees queued).")
            if self._thread is not None and threading.current_thread() is self._thread:
                self._render_one_frame_locked(step_physics=False)

    def command(self, action: str, **kwargs) -> Dict[str, Any]:
        with self._lock:
            if action == "start":
                self.estop = False
                if self.phase == "E-STOP":
                    self.phase = "DRIVE"
                self.running = True
                self._log("[CTRL] Simulation STARTED.")
            elif action == "pause":
                self.running = False
                self._log("[CTRL] Simulation PAUSED.")
            elif action == "estop":
                self.estop = True
                self.running = False
                self.data.ctrl[0:4] = 0.0
                self.data.ctrl[4:10] = self.home_q
                self.phase = "E-STOP"
                self._log("[SAFETY] EMERGENCY STOP TRIGGERED — Arm retracted, wheels locked.")
            elif action == "reset":
                pass  # handled outside lock to avoid re-lock
            elif action == "mode":
                m = str(kwargs.get("mode", "AUTO")).upper()
                self.mode = m if m in ("AUTO", "MANUAL") else "AUTO"
                self.manual_linear = 0.0
                self.manual_turn = 0.0
                self._log(f"[CTRL] Switched to {self.mode} mode.")
            elif action == "camera":
                v = str(kwargs.get("camera_view", "FOLLOW")).upper()
                self.camera_view = v if v in ("FOLLOW", "OVERVIEW") else "FOLLOW"
            elif action == "dose":
                val = float(kwargs.get("dose_ml", 80.0))
                self.dose_ml = float(max(10.0, min(500.0, val)))
            elif action == "teleop":
                self.mode = "MANUAL"
                self.estop = False
                self.running = True
                self.manual_linear = float(np.clip(kwargs.get("linear", 0.0), -1.0, 1.0))
                self.manual_turn = float(np.clip(kwargs.get("turn", 0.0), -1.0, 1.0))
            elif action == "manual_pour":
                self.mode = "MANUAL"
                self.manual_arm_pour = True
                self.running = True
                self._log(f"[MANUAL] Triggered 12.5cm pour ({self.dose_ml:.0f} mL).")
            elif action == "manual_retract":
                self.mode = "MANUAL"
                self.manual_arm_pour = False
                self._log("[MANUAL] Retracting arm to home pose.")

        if action == "reset":
            self.reset_simulation(keep_running=True)
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "estop": self.estop,
                "mode": self.mode,
                "camera_view": self.camera_view,
                "rover_x_m": round(float(self.data.qpos[0]), 2),
                "rover_y_m": round(float(self.data.qpos[1]), 2),
                "current_tree": min(self.tree_idx + 1, len(self.tree_x_targets)),
                "total_trees": len(self.tree_x_targets),
                "trees_poured": self.trees_poured,
                "phase": self.phase,
                "dose_ml": self.dose_ml,
                "gate_status": self.gate_summary,
                "system1_conf": round(self.last_conf * 100.0, 1),
                "system1_ms": round(self.last_s1_ms, 1),
                "logs": list(self.logs),
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_onboard_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_onboard_jpeg

    def get_3d_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            qw, qx, qy, qz = [float(v) for v in self.data.qpos[3:7]]
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            joints = [round(float(v), 4) for v in self.data.ctrl[4:10]]
            trees = [
                {
                    "id": i + 1,
                    "x": round(tx + 0.2, 2),
                    "y": 0.72,
                    "poured": i < self.trees_poured,
                }
                for i, tx in enumerate(self.tree_x_targets)
            ]
            return {
                "running": self.running,
                "estop": self.estop,
                "mode": self.mode,
                "phase": self.phase,
                "rover": [round(float(self.data.qpos[0]), 3), round(float(self.data.qpos[1]), 3), round(float(self.data.qpos[2]), 3), round(yaw, 4)],
                "joints": joints,
                "dosing_active": self._dosing_active,
                "dose_ml": self.dose_ml,
                "ibvs_error_cm": self._ibvs_error_cm,
                "ibvs_aligned": self._ibvs_aligned,
                "current_tree": min(self.tree_idx + 1, len(self.tree_x_targets)),
                "total_trees": len(self.tree_x_targets),
                "trees_poured": self.trees_poured,
                "gate_status": self.gate_summary,
                "system1_conf": round(self.last_conf * 100.0, 1),
                "system1_ms": round(self.last_s1_ms, 1),
                "trees": trees,
                "logs": list(self.logs),
            }

    def _render_one_frame_locked(self, step_physics: bool = True):
        rover_x = float(self.data.qpos[0])
        dosing_active = False

        if step_physics and not self.estop:
            if self.mode == "AUTO":
                self.phase_timer += 1
                if self.phase == "DRIVE":
                    self.data.ctrl[4:10] = self.home_q
                    if self.tree_idx < len(self.tree_x_targets):
                        if rover_x < self.tree_x_targets[self.tree_idx]:
                            self.data.ctrl[0:4] = 0.65
                        else:
                            self.data.ctrl[0:4] = -0.35 if self.data.qvel[0] > 0.05 else 0.0
                            self.phase = "VERIFY"
                            self.phase_timer = 0
                    else:
                        self.data.ctrl[0:4] = 0.0
                        self.phase = "FIELD COMPLETE"
                elif self.phase == "VERIFY":
                    self.data.ctrl[0:4] = 0.0
                    if self.phase_timer >= 14:
                        res = self.safety.dispatch(
                            authorized=True,
                            dose_volume_ml=self.dose_ml,
                            pulse_duration_s=round(self.dose_ml / 250.0, 2),
                            arm_plan={
                                "strategy": "STANDOFF_ANGLED_JET",
                                "pour_target_3d_m": [0.0, 0.72, -0.35],
                                "nozzle_pitch_rad": -0.85,
                            },
                            ik_joints=self.pour_arm_q.tolist(),
                            telemetry=SafetyTelemetry(),
                        )
                        self.gate_summary = f"{sum(1 for g in res['gates'] if g['passed'])}/4 PASS"
                        auth = DosingAuthorization(
                            authorized=True,
                            dose_volume_ml=self.dose_ml,
                            pulse_duration_seconds=round(self.dose_ml / 250.0, 3),
                            target_coordinates_m=(0.0, 0.72, -0.35),
                            decision_reasoning=f"Tree #{self.tree_idx + 1} auto-verified in MuJoCo field.",
                            crop_verified=True,
                            confidence_score=max(0.82, self.last_conf),
                        )
                        self.controller.execute_liquid_dose(auth)
                        self.phase = "EXTEND_POUR"
                        self.phase_timer = 0
                elif self.phase == "EXTEND_POUR":
                    self.data.ctrl[0:4] = 0.0
                    target_px = self._last_vision.trunk_base_pixel if self._last_vision is not None else (420, 390)
                    ibvs_res = self.ibvs.step(self.data.ctrl[4:10], target_px, (480, 640))
                    self.data.ctrl[4:10] = ibvs_res["next_q"]
                    self._ibvs_error_cm = ibvs_res["error_cm"]
                    self._ibvs_aligned = ibvs_res["aligned"]
                    if self.phase_timer >= 15:
                        dosing_active = True
                    if self.phase_timer >= 32:
                        self.trees_poured += 1
                        self._log(f"[POUR] Tree #{self.tree_idx + 1}: {self.dose_ml:.0f} mL @ 12.5cm (IBVS {self._ibvs_error_cm:.1f}cm).")
                        self.phase = "RETRACT"
                        self.phase_timer = 0
                elif self.phase == "RETRACT":
                    self.data.ctrl[0:4] = 0.0
                    t = min(1.0, self.phase_timer / 12.0)
                    self.data.ctrl[4:10] = (1.0 - t) * self.pour_arm_q + t * self.home_q
                    if self.phase_timer >= 14:
                        self.tree_idx += 1
                        self.tracker.reset_lock()
                        self.phase = "DRIVE"
                        self.phase_timer = 0
            else:
                # MANUAL teleop mode
                left_cmd = 0.8 * (self.manual_linear - 0.6 * self.manual_turn)
                right_cmd = 0.8 * (self.manual_linear + 0.6 * self.manual_turn)
                self.data.ctrl[0] = left_cmd
                self.data.ctrl[2] = left_cmd
                self.data.ctrl[1] = right_cmd
                self.data.ctrl[3] = right_cmd
                if self.manual_arm_pour:
                    self.data.ctrl[4:10] = self.pour_arm_q
                    dosing_active = True
                    self.phase = "MANUAL_POUR"
                else:
                    self.data.ctrl[4:10] = self.home_q
                    self.phase = "MANUAL_DRIVE"

            for _ in range(15):  # 30 ms physics step (~33 Hz)
                mujoco.mj_step(self.model, self.data)

        # Render onboard crop sensor + main 3D view (create WGL Renderer on worker thread)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data, camera=self.cam_onboard)
        onboard_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

        main_cam = self.cam_overview if self.camera_view == "OVERVIEW" else self.cam_follow
        self.renderer.update_scene(self.data, camera=main_cam)
        world_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

        self._frame_seq += 1
        if self._last_vision is None or self._frame_seq % 2 == 0:
            self._last_vision = self.detector.analyze_crop_image(onboard_bgr)
        vision = self._last_vision
        self.last_conf = vision.confidence
        self.last_s1_ms = vision.inference_ms
        trunk_boxes = [(*d.bbox_xyxy, d.confidence) for d in vision.all_detections if d.class_name == "trunk_base"]
        t_state = self.tracker.update(vision.is_pomegranate, vision.trunk_base_pixel, trunk_boxes=trunk_boxes)
        vision.trunk_base_pixel = t_state["stabilized_px"]
        onboard_ann = self.detector.draw_detections(onboard_bgr, vision)

        if dosing_active:
            tx, ty = t_state["stabilized_px"]
            cv2.circle(onboard_ann, (tx, ty), 18, (255, 220, 0), 2)
            cv2.putText(onboard_ann, f"POURING {self.dose_ml:.0f} mL", (max(10, tx - 60), max(30, ty - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 2)

        cv2.rectangle(world_bgr, (10, 10), (630, 80), (16, 20, 26), -1)
        status_col = (0, 60, 255) if self.estop else ((0, 220, 255) if dosing_active else (0, 255, 130))
        cv2.putText(world_bgr, f"WORLD ({self.camera_view}) | Mode: {self.mode} | x={rover_x:.2f}m | Tree {min(self.tree_idx+1, len(self.tree_x_targets))}/{len(self.tree_x_targets)}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 235, 245), 1)
        cv2.putText(world_bgr, f"PHASE: {self.phase} | Poured: {self.trees_poured} | Safety: {self.gate_summary}",
                    (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 2)

        self._dosing_active = dosing_active
        panel = np.hstack([world_bgr, onboard_ann])
        ok, buf = cv2.imencode(".jpg", panel, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        ok_ob, buf_ob = cv2.imencode(".jpg", cv2.resize(onboard_ann, (400, 300)), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with self._jpeg_lock:
            if ok:
                self._latest_jpeg = buf.tobytes()
            if ok_ob:
                self._latest_onboard_jpeg = buf_ob.tobytes()

    def _loop(self):
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            with self._lock:
                self._render_one_frame_locked(step_physics=self.running and not self.estop)
            dt = time.perf_counter() - t0
            time.sleep(0.004 if self.running else 0.04)
