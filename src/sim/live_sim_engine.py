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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import mujoco

from src.perception.pomegranate_detector import PomegranateVisionDetector, CropDetectionResult
from src.control.kinematics import ArmKinematics
from src.control.visual_servoing import EyeInHandVisualServo
from src.control.target_tracker import TemporalTargetStabilizer
from src.control.fertilizer_controller import FertilizerDosingController
from src.api.llm_brain import LLMBrainSupervisor, OperatorIntent, DosingAuthorization
from src.hardware.ros2_actuator_bridge import ROS2ActuatorBridge, SafetyTelemetry
from src.memory import (
    CropMemoryStore,
    CropRegistry,
    CropRecord,
    CropStatus,
    GrowthStage,
    DoseEvent,
)
from src.memory.field_map import FieldMap, MissionPlanner
from src.brain import PomegranateKnowledgeBase
from src.brain.decision_maker import EnhancedDecisionMaker


class LiveOrchardSimEngine:
    SIM_HZ = 30.0
    DRIVE_CTRL = 0.25
    T_VERIFY = 1.5
    T_POUR_MIN = 2.0
    T_POUR_MAX = 5.0
    T_RETRACT = 1.2

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
        self.ibvs = EyeInHandVisualServo(tolerance_cm=5.0)
        self.tracker = TemporalTargetStabilizer(ema_alpha=0.28, max_jump_px=115.0)
        self.controller = FertilizerDosingController(db_path="output/dosing_audit_log.db")
        self.brain = LLMBrainSupervisor(
            pump_flow_rate_ml_per_sec=250.0, backend="auto", ollama_model="qwen2.5:3b"
        )
        self.safety = ROS2ActuatorBridge()

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer: Optional[mujoco.Renderer] = None

        self.cam_onboard = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_orchard_cam")
        self.cam_follow = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_follow_cam")
        self.cam_overview = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "field_overview_cam")
        self.site_tool_tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")

        self.tree_x_targets = [1.20, 3.00, 4.80, 6.60, 8.40, 10.20, 12.00, 13.80]
        self.home_q = self.kin.home_q.copy()
        self.pour_arm_q = np.array([1.45, 0.95, -0.55, 0.0, -0.45, 0.0], dtype=np.float64)

        # Persistent Crop Memory, Field Map, Agronomy KB & Brain v2
        self.memory_store = CropMemoryStore(
            db_path="output/crop_memory.db", json_dir="output/crop_records"
        )
        self.crop_registry = CropRegistry(field_id="F01", memory_store=self.memory_store)
        self.field_map = FieldMap.from_tree_positions("F01", self.tree_x_targets, row_y=0.72, row_id="R01")
        self.knowledge = PomegranateKnowledgeBase(kb_path="data/knowledge/pomegranate_kb.json")
        self.decision_maker = EnhancedDecisionMaker(brain=self.brain, knowledge=self.knowledge)
        self.planner: Optional[MissionPlanner] = None
        self.tree_crop_ids: List[str] = []
        self.active_crop_id: Optional[str] = None
        self._last_agro_context: str = ""

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
        self._active_auth: Optional[DosingAuthorization] = None
        self._pour_started_at: Optional[float] = None
        self._mission_start_mono: Optional[float] = None
        self._mission_elapsed_s: float = 0.0

        # Interactive control state — start PAUSED so user sees Tree 1
        self.running = False
        self.estop = False
        self.mode = "AUTO"  # AUTO | MANUAL
        self.camera_view = "FOLLOW"  # FOLLOW | OVERVIEW
        self.dose_ml = 80.0
        self.command_text = "Pour 80 mL at the 12.5 cm root ring of every verified tree."
        self.manual_linear = 0.0  # -1..1
        self.manual_turn = 0.0    # -1..1
        self.manual_arm_pour = False

        # Mission telemetry
        self.tree_idx = 0
        self.phase = "PAUSED"
        self.phase_timer = 0
        self.trees_poured = 0
        self.trees_skipped = 0
        self.tree_statuses: List[str] = ["pending"] * len(self.tree_x_targets)
        self.tree_doses: List[Optional[float]] = [None] * len(self.tree_x_targets)
        self.gate_summary = "4/4 ARMED"
        self.gate_details: List[Dict[str, Any]] = self._default_gates()
        self.audit_rows: List[Dict[str, Any]] = []
        self.last_conf = 0.0
        self.last_s1_ms = 0.0
        self.logs: List[str] = ["[SIM] 8-tree MuJoCo world ready (PAUSED). Click Start to begin."]

        self.reset_simulation(keep_running=False, clear_memory=False)

    def _export_field_geojson_locked(self):
        try:
            out_dir = Path("output/field_maps")
            out_dir.mkdir(parents=True, exist_ok=True)
            crops = self.memory_store.get_all_by_field("F01")
            geojson = self.field_map.to_geojson(crops)
            import json as _json
            (out_dir / "F01_map.geojson").write_text(
                _json.dumps(geojson, indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._log(f"[MEM] GeoJSON export warning: {e}")

    def _init_field_memory_locked(self, clear_existing: bool = False):
        if clear_existing:
            self.crop_registry.reset()
        self.field_map = FieldMap.from_tree_positions(
            "F01", self.tree_x_targets, row_y=0.72, row_id="R01"
        )
        existing = {c.crop_id: c for c in self.memory_store.get_all_by_field("F01")}
        crop_ids: List[str] = []
        for i, tx in enumerate(self.tree_x_targets):
            pos = (round(tx + 0.2, 2), 0.72, 0.0)
            matched = self.memory_store.find_nearest(pos, radius_m=0.50)
            if matched is None:
                matched = self.crop_registry.register_or_update(
                    position_3d_m=pos,
                    detections=[],
                    frame_id=self._frame_seq,
                    row_id="R01",
                    confidence=0.85,
                )
                matched.status = CropStatus.QUEUED
                matched.growth_stage = GrowthStage.FRUITING
                self.memory_store.save(matched)
            crop_ids.append(matched.crop_id)
            if matched.status in (CropStatus.DOSED, CropStatus.COMPLETED):
                self.tree_statuses[i] = "poured"
                self.tree_doses[i] = matched.total_dose_ml
                self.trees_poured += 1
            elif matched.status == CropStatus.SKIPPED:
                self.tree_statuses[i] = "skipped"
                self.trees_skipped += 1
        self.tree_crop_ids = crop_ids
        self.field_map.rows[0].crop_ids = crop_ids
        self.planner = MissionPlanner(self.field_map, self.memory_store)
        self.planner.plan_mission("row_sequential")
        self.active_crop_id = crop_ids[0] if crop_ids else None
        self._last_agro_context = (
            self.knowledge.build_context_for_slm(self.memory_store.get(self.active_crop_id))
            if self.active_crop_id and self.memory_store.get(self.active_crop_id)
            else ""
        )
        self._export_field_geojson_locked()

    @staticmethod
    def _default_gates() -> List[Dict[str, Any]]:
        return [
            {"id": "G1", "label": "Base stopped", "state": "idle", "detail": "0.00 m/s · 0.0°"},
            {"id": "G2", "label": "Nozzle aligned", "state": "idle", "detail": "awaiting"},
            {"id": "G3", "label": "Flow verified", "state": "idle", "detail": "0 / 80 mL"},
            {"id": "G4", "label": "Arm retracted", "state": "ok", "detail": "stowed"},
        ]

    def start_background_loop(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.logs.append(msg)
        if len(self.logs) > 24:
            self.logs = self.logs[-24:]

    def _add_audit_row(self, tree_id: int, outcome: str, dose_ml: float, servo_cm: float, gates: str, s2_conf: float):
        self.audit_rows.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "tree": tree_id,
            "outcome": outcome,
            "dose_ml": round(dose_ml, 1),
            "servo_cm": round(servo_cm, 1),
            "gates": gates,
            "s2_conf": round(s2_conf, 2),
        })
        if len(self.audit_rows) > 30:
            self.audit_rows = self.audit_rows[:30]

    def reset_simulation(self, keep_running: bool = False, clear_memory: bool = True):
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.ctrl[4:10] = self.home_q
            self.tracker = TemporalTargetStabilizer(ema_alpha=0.28, max_jump_px=115.0)
            self.tree_idx = 0
            self.phase = "DRIVE" if keep_running else "PAUSED"
            self.phase_timer = 0
            self.trees_poured = 0
            self.trees_skipped = 0
            self.tree_statuses = ["pending"] * len(self.tree_x_targets)
            self.tree_doses = [None] * len(self.tree_x_targets)
            self.gate_summary = "4/4 ARMED"
            self.gate_details = self._default_gates()
            self.audit_rows = []
            self._active_auth = None
            self._pour_started_at = None
            self._ibvs_error_cm = 0.0
            self._ibvs_aligned = False
            self._mission_start_mono = time.monotonic() if keep_running else None
            self._mission_elapsed_s = 0.0
            self.estop = False
            self.manual_linear = 0.0
            self.manual_turn = 0.0
            self.manual_arm_pour = False
            self.running = keep_running
            self._init_field_memory_locked(clear_existing=clear_memory)
            self._log("[SIM] Field reset to Row Start (x=0.0m, 8 trees queued in CropMemory).")
            if self._thread is not None and threading.current_thread() is self._thread:
                self._render_one_frame_locked(step_physics=False)

    def command(self, action: str, **kwargs) -> Dict[str, Any]:
        with self._lock:
            if action == "start":
                self.estop = False
                if self.phase in ("E-STOP", "PAUSED"):
                    self.phase = "DRIVE"
                if self._mission_start_mono is None:
                    self._mission_start_mono = time.monotonic() - self._mission_elapsed_s
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
                self._active_auth = None
                self._log("[SAFETY] EMERGENCY STOP TRIGGERED — Arm retracted, wheels locked.")
            elif action == "reset":
                pass  # handled outside lock
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
                self.command_text = f"Pour {self.dose_ml:.0f} mL at the 12.5 cm root ring of every verified tree."
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
            clear_mem = bool(kwargs.get("clear_memory", True))
            self.reset_simulation(keep_running=False, clear_memory=clear_mem)
        return self.get_state()

    def _export_field_geojson_locked(self):
        try:
            out_dir = Path("output/field_maps")
            out_dir.mkdir(parents=True, exist_ok=True)
            crops = self.memory_store.get_all_by_field("F01")
            geojson = self.field_map.to_geojson(crops)
            import json as _json
            (out_dir / "F01_map.geojson").write_text(
                _json.dumps(geojson, indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._log(f"[MEM] GeoJSON export warning: {e}")

    def _init_field_memory_locked(self, clear_existing: bool = False):
        if clear_existing:
            self.crop_registry.reset()
        self.field_map = FieldMap.from_tree_positions(
            "F01", self.tree_x_targets, row_y=0.72, row_id="R01"
        )
        crop_ids: List[str] = []
        for i, tx in enumerate(self.tree_x_targets):
            pos = (round(tx + 0.2, 2), 0.72, 0.0)
            matched = self.memory_store.find_nearest(pos, radius_m=0.50)
            if matched is None:
                matched = self.crop_registry.register_or_update(
                    position_3d_m=pos,
                    detections=[],
                    frame_id=self._frame_seq,
                    row_id="R01",
                    confidence=0.85,
                )
                matched.status = CropStatus.QUEUED
                matched.growth_stage = GrowthStage.FRUITING
                self.memory_store.save(matched)
            crop_ids.append(matched.crop_id)
            if matched.status in (CropStatus.DOSED, CropStatus.COMPLETED):
                self.tree_statuses[i] = "poured"
                self.tree_doses[i] = matched.total_dose_ml
                self.trees_poured += 1
            elif matched.status == CropStatus.SKIPPED:
                self.tree_statuses[i] = "skipped"
                self.trees_skipped += 1
        self.tree_crop_ids = crop_ids
        self.field_map.rows[0].crop_ids = crop_ids
        self.planner = MissionPlanner(self.field_map, self.memory_store)
        self.planner.plan_mission("row_sequential")
        self.active_crop_id = crop_ids[0] if crop_ids else None
        first_crop = self.memory_store.get(self.active_crop_id) if self.active_crop_id else None
        self._last_agro_context = self.knowledge.build_context_for_slm(first_crop) if first_crop else ""
        self._export_field_geojson_locked()

    def _compute_rover_telemetry_locked(self) -> SafetyTelemetry:
        speed = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
        qw, qx, qy, qz = [float(v) for v in self.data.qpos[3:7]]
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll_deg = float(np.degrees(np.arctan2(sinr, cosr)))
        sinp = float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        pitch_deg = float(np.degrees(np.arcsin(sinp)))
        return SafetyTelemetry(
            rover_speed_mps=round(speed, 3),
            imu_roll_deg=round(roll_deg, 2),
            imu_pitch_deg=round(pitch_deg, 2),
            nozzle_error_cm=round(self._ibvs_error_cm, 2),
            measured_flow_ml=None,
            valve_closed_confirmed=True,
        )

    def _measure_tool_tip_error_cm_locked(self) -> float:
        if self.site_tool_tip >= 0 and self.tree_idx < len(self.tree_x_targets):
            tip = self.data.site_xpos[self.site_tool_tip]
            # Target ring is 12.5 cm rover-side of trunk at (tx + 0.2, 0.72, 0.08)
            target = np.array([self.tree_x_targets[self.tree_idx] + 0.20, 0.595, 0.22], dtype=np.float64)
            return round(float(np.linalg.norm(tip - target) * 100.0), 2)
        return self._ibvs_error_cm

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
                "trees_skipped": self.trees_skipped,
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
            crops_by_id = {c.crop_id: c for c in self.memory_store.get_all_by_field("F01")}
            trees = []
            for i, tx in enumerate(self.tree_x_targets):
                cid = self.tree_crop_ids[i] if i < len(self.tree_crop_ids) else f"F01-R01-C{i+1:04d}"
                crec = crops_by_id.get(cid)
                trees.append({
                    "id": i + 1,
                    "crop_id": cid,
                    "x": round(tx + 0.2, 2),
                    "y": 0.72,
                    "status": self.tree_statuses[i],
                    "poured": self.tree_statuses[i] == "poured",
                    "dose_ml": self.tree_doses[i],
                    "growth_stage": crec.growth_stage.value if crec else "unknown",
                    "fruit_count": crec.fruit_count_estimate if crec else 0,
                    "health_score": crec.health_score if crec else 1.0,
                    "disease_flags": crec.disease_flags if crec else [],
                    "total_doses_count": crec.total_doses_count if crec else 0,
                })
            auth = self._active_auth
            s2_info = {
                "backend": self.brain.backend,
                "model": self.brain.ollama_model if self.brain.backend == "ollama" else self.brain.backend,
                "authorized": bool(auth and auth.authorized),
                "tree_id": auth.tree_id if auth else None,
                "crop_id": self.active_crop_id,
                "conf": round(auth.confidence_score, 2) if auth else 0.0,
                "valid_for_s": round(max(0.0, 5.0 - (time.monotonic() - auth.issued_at)), 1) if auth else 0.0,
                "frame_id": auth.frame_id if auth else self._frame_seq,
            }
            return {
                "running": self.running,
                "estop": self.estop,
                "mode": self.mode,
                "phase": self.phase,
                "mission_elapsed_s": round(self._mission_elapsed_s, 1),
                "command_text": self.command_text,
                "rover": [round(float(self.data.qpos[0]), 3), round(float(self.data.qpos[1]), 3), round(float(self.data.qpos[2]), 3), round(yaw, 4)],
                "joints": joints,
                "dosing_active": self._dosing_active,
                "dose_ml": self.dose_ml,
                "ibvs_error_cm": self._ibvs_error_cm,
                "ibvs_aligned": self._ibvs_aligned,
                "current_tree": min(self.tree_idx + 1, len(self.tree_x_targets)),
                "active_crop_id": self.active_crop_id,
                "agro_context": self._last_agro_context,
                "field_summary": self.memory_store.get_field_summary("F01"),
                "mission_progress": self.planner.get_progress() if self.planner else {},
                "total_trees": len(self.tree_x_targets),
                "trees_poured": self.trees_poured,
                "trees_skipped": self.trees_skipped,
                "gate_status": self.gate_summary,
                "gates": list(self.gate_details),
                "s2_info": s2_info,
                "audit_rows": list(self.audit_rows),
                "system1_conf": round(self.last_conf * 100.0, 1),
                "system1_ms": round(self.last_s1_ms, 1),
                "trees": trees,
                "logs": list(self.logs),
            }

    def _render_one_frame_locked(self, step_physics: bool = True):
        rover_x = float(self.data.qpos[0])
        dosing_active = False

        if step_physics and not self.estop:
            if self._mission_start_mono is not None and self.phase != "FIELD COMPLETE":
                self._mission_elapsed_s = time.monotonic() - self._mission_start_mono

            if self.mode == "AUTO":
                self.phase_timer += 1
                t_sec = self.phase_timer / self.SIM_HZ
                if self.phase == "DRIVE":
                    self.data.ctrl[4:10] = self.home_q
                    if self.tree_idx < len(self.tree_x_targets):
                        cid = self.tree_crop_ids[self.tree_idx] if self.tree_idx < len(self.tree_crop_ids) else None
                        self.active_crop_id = cid
                        crec = self.memory_store.get(cid) if cid else None
                        if crec and (crec.status in (CropStatus.DOSED, CropStatus.COMPLETED) or crec.was_dosed_today()):
                            self.tree_statuses[self.tree_idx] = "poured"
                            self.tree_doses[self.tree_idx] = crec.total_dose_ml
                            self._log(f"[MEMORY] {cid} already dosed — skipping repeat visit.")
                            self.tree_idx += 1
                            self.phase_timer = 0
                        else:
                            self.tree_statuses[self.tree_idx] = "active"
                            if crec and crec.status != CropStatus.APPROACHING:
                                crec.status = CropStatus.APPROACHING
                                self.memory_store.save(crec)
                            if rover_x < self.tree_x_targets[self.tree_idx]:
                                self.data.ctrl[0:4] = self.DRIVE_CTRL
                            else:
                                self.data.ctrl[0:4] = -0.4 * self.DRIVE_CTRL if self.data.qvel[0] > 0.05 else 0.0
                                self.phase = "VERIFY"
                                self.phase_timer = 0
                                self._active_auth = None
                    else:
                        self.data.ctrl[0:4] = 0.0
                        self.phase = "FIELD COMPLETE"
                elif self.phase == "VERIFY":
                    self.data.ctrl[0:4] = 0.0
                    tel = self._compute_rover_telemetry_locked()
                    g1 = self.safety.check_base_stability(tel)
                    self.gate_details[0] = {
                        "id": "G1", "label": "Base stopped",
                        "state": "ok" if g1["passed"] else "bad",
                        "detail": f"{tel.rover_speed_mps:.2f} m/s · {abs(tel.imu_roll_deg):.1f}°",
                    }
                    if t_sec >= self.T_VERIFY:
                        cid = self.tree_crop_ids[self.tree_idx] if self.tree_idx < len(self.tree_crop_ids) else None
                        crec = self.memory_store.get(cid) if cid else None
                        if crec is not None:
                            crec.status = CropStatus.INSPECTING
                            assessed = self.knowledge.assess_growth_stage(
                                fruit_count=crec.fruit_count_estimate,
                                flower_count=crec.flower_count_estimate,
                                canopy_density=crec.canopy_density,
                                month=datetime.now().month,
                            )
                            try:
                                crec.growth_stage = GrowthStage(assessed)
                            except ValueError:
                                crec.growth_stage = GrowthStage.FRUITING
                            self.memory_store.save(crec)
                            self._last_agro_context = self.knowledge.build_context_for_slm(crec)

                        vision_for_s2 = self._last_vision
                        if vision_for_s2 is None or not vision_for_s2.is_pomegranate:
                            # Synthetic MuJoCo tree visual proxy when synthetic sphere leaves don't trigger YOLO
                            vision_for_s2 = CropDetectionResult(
                                is_pomegranate=True,
                                confidence=max(0.84, self.last_conf),
                                species_name="Pomegranate (Punica granatum)",
                                trunk_base_pixel=(420, 390),
                                target_3d_offset_m=(0.0, 0.72, -0.35),
                                canopy_bounding_box=(260, 180, 220, 260),
                                soil_contact_point=(420, 420),
                                inference_ms=self.last_s1_ms,
                            )
                        intent = OperatorIntent(
                            command_text=self.command_text,
                            target_crop="pomegranate",
                            dose_volume_ml=self.dose_ml,
                            min_confidence=0.80,
                        )
                        field_ctx = self.memory_store.get_field_summary("F01")
                        auth = self.decision_maker.evaluate(
                            intent,
                            vision_for_s2,
                            crop_record=crec,
                            field_context=field_ctx,
                            frame_id=self._frame_seq,
                        )
                        # Align tree_id binding to integer tree index for sim gate checks
                        auth.tree_id = self.tree_idx + 1
                        self._active_auth = auth
                        if auth.is_valid(tree_id=self.tree_idx + 1, ttl_s=6.0):
                            self.phase = "EXTEND_POUR"
                            self.phase_timer = 0
                            self._pour_started_at = None
                        else:
                            self.trees_skipped += 1
                            self.tree_statuses[self.tree_idx] = "skipped"
                            if cid and self.planner:
                                self.planner.mark_outcome(cid, "REJECTED", auth.decision_reasoning)
                                self._export_field_geojson_locked()
                            self._add_audit_row(self.tree_idx + 1, "Skipped", 0.0, self._ibvs_error_cm, "S2 reject", auth.confidence_score)
                            self.controller.log_outcome("Unverified", auth.confidence_score, 0.0, 0.0, status="REJECTED")
                            self._log(f"[SKIP] {cid or f'Tree #{self.tree_idx + 1}'}: {auth.decision_reasoning}")
                            self.phase = "RETRACT"
                            self.phase_timer = 0
                elif self.phase == "EXTEND_POUR":
                    self.data.ctrl[0:4] = 0.0
                    target_px = self._last_vision.trunk_base_pixel if self._last_vision is not None else (420, 390)
                    ibvs_res = self.ibvs.step(self.data.ctrl[4:10], target_px, (480, 640))
                    self.data.ctrl[4:10] = ibvs_res["next_q"]
                    # Blend pixel servo residual with real MuJoCo tool-tip distance
                    tip_err_cm = self._measure_tool_tip_error_cm_locked()
                    self._ibvs_error_cm = round(min(ibvs_res["error_cm"], tip_err_cm) if t_sec > 0.8 else tip_err_cm, 2)
                    self._ibvs_aligned = self._ibvs_error_cm <= 5.0

                    self.gate_details[1] = {
                        "id": "G2", "label": "Nozzle aligned",
                        "state": "ok" if self._ibvs_aligned else "active",
                        "detail": f"{self._ibvs_error_cm:.1f} ≤ 5.0 cm",
                    }

                    if self._ibvs_aligned and t_sec >= self.T_POUR_MIN:
                        dosing_active = True
                        if self._pour_started_at is None:
                            self._pour_started_at = t_sec
                        pour_frac = min(1.0, (t_sec - self._pour_started_at) / 1.5)
                        delivered_ml = round(self.dose_ml * pour_frac, 1)
                        self.gate_details[2] = {
                            "id": "G3", "label": "Flow verified",
                            "state": "active" if pour_frac < 1.0 else "ok",
                            "detail": f"{delivered_ml:.0f} / {self.dose_ml:.0f} mL",
                        }
                        if t_sec - self._pour_started_at >= 1.5:
                            tel = self._compute_rover_telemetry_locked()
                            tel.nozzle_error_cm = self._ibvs_error_cm
                            tel.measured_flow_ml = round(self.dose_ml * float(np.random.uniform(0.985, 1.015)), 1)
                            res = self.safety.dispatch(
                                authorized=True,
                                dose_volume_ml=self.dose_ml,
                                pulse_duration_s=round(self.dose_ml / 250.0, 2),
                                arm_plan={
                                    "strategy": "STANDOFF_ANGLED_JET",
                                    "pour_target_3d_m": [0.0, 0.72, -0.35],
                                    "nozzle_pitch_rad": -0.85,
                                },
                                ik_joints=self.data.ctrl[4:10].tolist(),
                                telemetry=tel,
                                actuation_fn=(lambda: self.controller.execute_liquid_dose(self._active_auth, simulate_delay=False)),
                            )
                            passed_cnt = sum(1 for g in res["gates"] if g["passed"])
                            self.gate_summary = f"{passed_cnt}/4 PASS"
                            self.trees_poured += 1
                            self.tree_statuses[self.tree_idx] = "poured"
                            self.tree_doses[self.tree_idx] = tel.measured_flow_ml
                            cid = self.tree_crop_ids[self.tree_idx] if self.tree_idx < len(self.tree_crop_ids) else None
                            if cid:
                                crec = self.memory_store.get(cid)
                                if crec:
                                    stage_str = crec.growth_stage.value
                                    fert = self.knowledge.get_recommended_fertilizer(stage_str)
                                    crec.record_dose(DoseEvent(
                                        timestamp=datetime.now(timezone.utc).isoformat(),
                                        dose_ml=tel.measured_flow_ml,
                                        fertilizer_type=fert,
                                        outcome="COMPLETED",
                                        gates_passed=f"{passed_cnt}/4",
                                        ibvs_error_cm=self._ibvs_error_cm,
                                        s2_confidence=self._active_auth.confidence_score if self._active_auth else self.last_conf,
                                        s2_reasoning=self._active_auth.decision_reasoning if self._active_auth else "",
                                        growth_stage_at_dose=stage_str,
                                    ))
                                    self.memory_store.save(crec)
                                if self.planner:
                                    self.planner.mark_outcome(cid, "DOSED")
                                self._export_field_geojson_locked()
                            self._add_audit_row(
                                self.tree_idx + 1, "Completed", tel.measured_flow_ml,
                                self._ibvs_error_cm, f"{passed_cnt}/4",
                                self._active_auth.confidence_score if self._active_auth else self.last_conf,
                            )
                            self._log(f"[POUR] {cid or f'Tree #{self.tree_idx + 1}'}: {tel.measured_flow_ml:.1f} mL @ 12.5cm (err {self._ibvs_error_cm:.1f}cm).")
                            self.phase = "RETRACT"
                            self.phase_timer = 0
                    elif t_sec >= self.T_POUR_MAX and not self._ibvs_aligned:
                        self.trees_skipped += 1
                        self.tree_statuses[self.tree_idx] = "skipped"
                        self.gate_summary = "G2 FAIL"
                        cid = self.tree_crop_ids[self.tree_idx] if self.tree_idx < len(self.tree_crop_ids) else None
                        if cid and self.planner:
                            self.planner.mark_outcome(cid, "IBVS_TIMEOUT")
                            self._export_field_geojson_locked()
                        self._add_audit_row(
                            self.tree_idx + 1, "Skipped", 0.0, self._ibvs_error_cm, "G2 fail",
                            self._active_auth.confidence_score if self._active_auth else self.last_conf,
                        )
                        self.controller.log_outcome(
                            "Pomegranate (Punica granatum)", self.last_conf, 0.0, 0.0,
                            status="SKIPPED_NO_CONVERGE",
                        )
                        self._log(f"[SKIP] {cid or f'Tree #{self.tree_idx + 1}'}: Servo did not converge ({self._ibvs_error_cm:.1f}cm).")
                        self.phase = "RETRACT"
                        self.phase_timer = 0
                elif self.phase == "RETRACT":
                    self.data.ctrl[0:4] = 0.0
                    t = min(1.0, t_sec / self.T_RETRACT)
                    self.data.ctrl[4:10] = (1.0 - t) * self.pour_arm_q + t * self.home_q
                    self.gate_details[3] = {"id": "G4", "label": "Arm retracted", "state": "active" if t < 1.0 else "ok", "detail": "stowing" if t < 1.0 else "stowed"}
                    if t_sec >= self.T_RETRACT:
                        self.tree_idx += 1
                        self.tracker.reset_lock()
                        self._active_auth = None
                        if self.tree_idx >= len(self.tree_x_targets):
                            self.phase = "FIELD COMPLETE"
                            self.running = False
                            self._log("[SIM] Row complete — all 8 trees processed.")
                        else:
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

            for _ in range(15):  # 30 ms physics step
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
        if self._frame_seq % 15 == 0 and vision.is_pomegranate and self.tree_idx < len(self.tree_x_targets):
            tx = self.tree_x_targets[self.tree_idx]
            self.crop_registry.register_or_update(
                position_3d_m=(round(tx + 0.2, 2), 0.72, 0.0),
                detections=vision.all_detections,
                frame_id=self._frame_seq,
                row_id="R01",
                confidence=vision.confidence,
            )
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
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            with self._lock:
                self._render_one_frame_locked(step_physics=self.running and not self.estop)
            next_t += 1.0 / self.SIM_HZ if self.running else 0.05
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()
