"""
FastAPI Mission Control & Voice/Text Command Server
====================================================
Production API layer for the Physical AI Dual-Brain:
  - System 1: YOLOv8s (models/best.pt)
  - 3D Lifting: Depth-Anything-V2-Small INT8 ONNX
  - System 2: Ollama qwen2.5:3b (with 0.80 safety floor)
  - Kinematics: 6-DOF ArmKinematics IK solver
  - Actuation: FertilizerDosingController + SQLite audit log
"""

import base64
import sqlite3
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.perception.depth_localizer import MonocularDepthLocalizer
from src.api.llm_brain import LLMBrainSupervisor, OperatorIntent
from src.control.kinematics import ArmKinematics
from src.control.fertilizer_controller import FertilizerDosingController
from src.hardware.ros2_actuator_bridge import ROS2ActuatorBridge

app = FastAPI(
    title="Physical AI Pomegranate Dual-Brain API",
    version="2.0.0",
    description="System 1 (YOLOv8s) + 3D Depth + System 2 (Qwen2.5-3B) Mission Control",
)

# Lazy-loaded singletons so server starts fast and reuses GPU/ONNX sessions
_state = {}


def get_components():
    if "detector" not in _state:
        _state["detector"] = PomegranateVisionDetector(
            weights_path="models/best.pt", confidence_threshold=0.70
        )
        _state["depth"] = MonocularDepthLocalizer()
        _state["brain"] = LLMBrainSupervisor(
            pump_flow_rate_ml_per_sec=250.0, backend="auto", ollama_model="qwen2.5:3b"
        )
        _state["kinematics"] = ArmKinematics()
        _state["controller"] = FertilizerDosingController(db_path="output/dosing_audit_log.db")
        _state["hw_bridge"] = ROS2ActuatorBridge()
        _state["active_intent"] = OperatorIntent(
            command_text="Default: Apply 80 mL bio-fertilizer to pomegranate base.",
            dose_volume_ml=80.0,
            min_confidence=0.80,
        )
    return _state


class CommandRequest(BaseModel):
    command_text: str


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = Path("src/api/static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return html_path.read_text(encoding="utf-8")


@app.get("/sample-images")
def list_sample_images():
    samples = []
    for sub in ("pomegranate", "negative"):
        folder = Path("data/real_test_samples") / sub
        if folder.exists():
            for p in sorted(folder.glob("*.jpg"))[:4]:
                samples.append({"label": f"{sub}/{p.name}", "path": p.as_posix()})
    return {"samples": samples}


@app.get("/sample-file")
def get_sample_file(path: str):
    p = Path(path).resolve()
    allowed = Path("data/real_test_samples").resolve()
    if not str(p).startswith(str(allowed)) or not p.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(str(p))


@app.get("/health")
def health():
    s = get_components()
    return {
        "status": "ok",
        "system1_backend": s["detector"].backend,
        "system1_device": str(s["detector"].device),
        "system2_backend": s["brain"].backend,
        "system2_model": s["brain"].ollama_model,
        "depth_model": s["depth"].onnx_file,
    }


@app.post("/command")
def set_operator_command(req: CommandRequest):
    s = get_components()
    intent = s["brain"].parse_operator_command(req.command_text)
    s["active_intent"] = intent
    return {"status": "updated", "intent": intent.model_dump()}


@app.post("/inspect-and-authorize")
async def inspect_and_authorize(
    file: UploadFile = File(...),
    execute_actuation: bool = Form(default=True),
    use_depth_3d: bool = Form(default=True),
    include_images: bool = Form(default=False),
):
    s = get_components()
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vision = s["detector"].analyze_crop_image(frame)
    depth_ms = 0.0
    norm_disp = None
    if use_depth_3d and vision.is_pomegranate:
        norm_disp, depth_ms = s["depth"].estimate_depth_map(frame)
        vision.target_3d_offset_m = s["depth"].pixel_to_3d(vision.trunk_base_pixel, norm_disp)

    decision = s["brain"].evaluate_dosing_decision(s["active_intent"], vision)

    ik_joints: Optional[List[float]] = None
    actuation = None
    arm_plan = None
    if decision.authorized:
        # 12.5 cm rover-side pour ring + low-branch obstacle check
        pour_3d = s["kinematics"].compute_precision_pour_target(decision.target_coordinates_m, ring_offset_m=0.125)
        obs_info = (
            s["kinematics"].analyze_low_branch_obstacles(vision.all_detections, norm_disp, s["depth"])
            if norm_disp is not None
            else {"min_branch_height_m": 0.60, "canopy_front_y_m": 1.20, "low_branch_risk": False}
        )
        plan = s["kinematics"].plan_collision_free_approach(np.array([0.0, 0.45, -0.25]), pour_3d, obs_info)
        ee = plan["target_end_effector_m"]
        arm_target = np.array([min(0.65, max(0.25, ee[1] * 0.45)), np.clip(ee[0], -0.35, 0.35), -0.15])
        ik_joints = [round(float(q), 4) for q in s["kinematics"].solve_ik_target(arm_target, target_pitch=plan["nozzle_pitch_rad"])]
        arm_plan = {
            "strategy": plan["strategy"],
            "nozzle_pitch_rad": plan["nozzle_pitch_rad"],
            "pour_target_3d_m": plan["pour_target_m"],
            "obstacles": obs_info,
        }
        if execute_actuation:
            actuation = s["controller"].execute_liquid_dose(decision)

    from dataclasses import asdict
    hw_frame = asdict(s["hw_bridge"].dispatch(
        authorized=decision.authorized,
        dose_volume_ml=decision.dose_volume_ml,
        pulse_duration_s=decision.pulse_duration_seconds,
        arm_plan=arm_plan,
        ik_joints=ik_joints,
    ))

    resp = {
        "vision": {
            "is_pomegranate": vision.is_pomegranate,
            "confidence": vision.confidence,
            "num_detections": len(vision.all_detections),
            "trunk_base_pixel": list(vision.trunk_base_pixel),
            "target_3d_offset_m": list(vision.target_3d_offset_m),
            "system1_ms": vision.inference_ms,
            "depth_ms": depth_ms,
        },
        "decision": decision.model_dump(),
        "arm_plan": arm_plan,
        "ik_joint_angles_rad": ik_joints,
        "hardware_frame": hw_frame,
        "actuation": actuation,
    }

    if include_images:
        ann = s["detector"].draw_detections(frame, vision)
        _, buf = cv2.imencode(".jpg", ann)
        resp["annotated_rgb_b64"] = base64.b64encode(buf).decode("ascii")
        if norm_disp is not None:
            dcol = s["depth"].colorize_depth(norm_disp)
            cv2.drawMarker(dcol, vision.trunk_base_pixel, (0, 255, 0), cv2.MARKER_CROSS, 24, 3)
            _, dbuf = cv2.imencode(".jpg", dcol)
            resp["depth_heatmap_b64"] = base64.b64encode(dbuf).decode("ascii")

    return resp


@app.get("/audit-log")
def get_audit_log(limit: int = 10):
    db_path = Path("output/dosing_audit_log.db")
    if not db_path.exists():
        return {"records": []}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM dosing_transactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"records": [dict(r) for r in rows]}
