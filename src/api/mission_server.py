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

import asyncio
import base64
from contextlib import closing
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Literal
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.perception.depth_localizer import MonocularDepthLocalizer
from src.api.llm_brain import LLMBrainSupervisor, OperatorIntent
from src.control.kinematics import ArmKinematics
from src.control.fertilizer_controller import FertilizerDosingController
from src.hardware.ros2_actuator_bridge import ROS2ActuatorBridge, SafetyTelemetry
from src.sim.live_sim_engine import LiveOrchardSimEngine
from src.memory import CropMemoryStore, CropStatus
from src.memory.field_map import FieldMap, MissionPlanner
from src.brain import PomegranateKnowledgeBase

app = FastAPI(
    title="Physical AI Pomegranate Dual-Brain API",
    version="2.0.0",
    description="System 1 (YOLOv8s) + 3D Depth + System 2 (Qwen2.5-3B) Mission Control",
)

# Lazy-loaded singletons so server starts fast and reuses GPU/ONNX sessions
_state = {}


def get_components():
    if "detector" not in _state:
        weights = (
            "models/phase3/best_phase3.pt"
            if Path("models/phase3/best_phase3.pt").exists()
            else "models/best.pt"
        )
        _state["detector"] = PomegranateVisionDetector(
            weights_path=weights, confidence_threshold=0.70, nms_conf=0.22
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
    if not p.is_relative_to(allowed) or not p.exists():
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
def inspect_and_authorize(
    file: UploadFile = File(...),
    execute_actuation: bool = Form(default=True),
    use_depth_3d: bool = Form(default=True),
    include_images: bool = Form(default=False),
    imu_roll_deg: float = Form(default=0.8),
    imu_pitch_deg: float = Form(default=-0.5),
    nozzle_error_cm: float = Form(default=1.4),
    measured_flow_ml: Optional[float] = Form(default=None),
    valve_closed_confirmed: bool = Form(default=True),
):
    s = get_components()
    raw = file.file.read()
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

    tel = SafetyTelemetry(
        rover_speed_mps=0.0,
        imu_roll_deg=imu_roll_deg,
        imu_pitch_deg=imu_pitch_deg,
        nozzle_error_cm=nozzle_error_cm,
        measured_flow_ml=measured_flow_ml,
        valve_closed_confirmed=valve_closed_confirmed,
    )
    actuation_fn = (
        (lambda: s["controller"].execute_liquid_dose(decision))
        if (decision.authorized and execute_actuation)
        else None
    )
    bridge_result = s["hw_bridge"].dispatch(
        authorized=decision.authorized,
        dose_volume_ml=decision.dose_volume_ml,
        pulse_duration_s=decision.pulse_duration_seconds,
        arm_plan=arm_plan,
        ik_joints=ik_joints,
        telemetry=tel,
        actuation_fn=actuation_fn,
    )

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
        "hardware_frame": bridge_result["frame"],
        "safety_gates": bridge_result["gates"],
        "safe_abort": bridge_result["safe_abort"],
        "actuation": bridge_result["actuation"],
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
def get_audit_log(limit: int = Query(default=10, ge=1, le=500)):
    db_path = Path("output/dosing_audit_log.db")
    if not db_path.exists():
        return {"records": []}
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM dosing_transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"records": [dict(r) for r in rows]}


# ═══════════════════════════════════════════════════════════════
#  LIVE FULL-FIELD 3D MUJOCO SIMULATION WEB COCKPIT (/sim)
# ═══════════════════════════════════════════════════════════════

def get_sim_engine() -> LiveOrchardSimEngine:
    if "sim_engine" not in _state:
        eng = LiveOrchardSimEngine()
        eng.start_background_loop()
        _state["sim_engine"] = eng
    return _state["sim_engine"]


class SimControlRequest(BaseModel):
    action: Literal[
        "start", "pause", "estop", "reset", "mode", "camera",
        "dose", "teleop", "manual_pour", "manual_retract"
    ]
    mode: Optional[str] = None
    camera_view: Optional[str] = None
    dose_ml: Optional[float] = Field(default=None, ge=10.0, le=500.0)
    linear: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0)
    turn: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0)


@app.get("/sim", response_class=HTMLResponse)
def sim_cockpit():
    html_path = Path("src/api/static/sim.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Simulation cockpit not found")
    return html_path.read_text(encoding="utf-8")


@app.get("/sim/state")
def sim_state():
    return get_sim_engine().get_state()


@app.get("/sim/telemetry")
def sim_telemetry():
    return get_sim_engine().get_3d_telemetry()


@app.get("/sim/onboard_stream")
async def sim_onboard_stream(request: Request):
    eng = get_sim_engine()

    async def _gen():
        last_jpg = None
        while not await request.is_disconnected():
            jpg = eng.get_onboard_jpeg()
            if jpg is not None and jpg is not last_jpg:
                last_jpg = jpg
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/sim/control")
def sim_control(req: SimControlRequest):
    eng = get_sim_engine()
    return eng.command(
        req.action,
        mode=req.mode,
        camera_view=req.camera_view,
        dose_ml=req.dose_ml,
        linear=req.linear,
        turn=req.turn,
    )


@app.get("/sim/stream")
async def sim_stream(request: Request):
    eng = get_sim_engine()

    async def _gen():
        last_jpg = None
        while not await request.is_disconnected():
            jpg = eng.get_latest_jpeg()
            if jpg is not None and jpg is not last_jpg:
                last_jpg = jpg
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(0.015)

    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/sim/assets/{filename}")
def sim_asset(filename: str):
    if filename not in ("orchard_turkey_web.ply", "orchard_3d_meta.json"):
        raise HTTPException(status_code=404, detail="Asset not found")
    p = Path("output") / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Asset not generated yet")
    return FileResponse(str(p))


# ═══════════════════════════════════════════════════════════════
#  PERSISTENT CROP MEMORY, FIELD MAP & AGRONOMY KNOWLEDGE API
# ═══════════════════════════════════════════════════════════════

def _get_memory_and_kb():
    if "sim_engine" in _state:
        eng = _state["sim_engine"]
        return eng.memory_store, eng.field_map, eng.planner, eng.knowledge
    if "memory_store" not in _state:
        _state["memory_store"] = CropMemoryStore(
            db_path="output/crop_memory.db", json_dir="output/crop_records"
        )
        _state["knowledge_kb"] = PomegranateKnowledgeBase(
            kb_path="data/knowledge/pomegranate_kb.json"
        )
        _state["field_map"] = FieldMap.from_tree_positions(
            "F01", [1.20, 3.00, 4.80, 6.60, 8.40, 10.20, 12.00, 13.80]
        )
        _state["planner"] = MissionPlanner(_state["field_map"], _state["memory_store"])
        _state["planner"].plan_mission("row_sequential")
    return _state["memory_store"], _state["field_map"], _state["planner"], _state["knowledge_kb"]


class MissionPlanRequest(BaseModel):
    strategy: Literal["row_sequential", "priority_health"] = "row_sequential"


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    crop_id: Optional[str] = None


@app.get("/field-map")
def get_field_map(field_id: str = "F01"):
    mem, fmap, _, _ = _get_memory_and_kb()
    crops = mem.get_all_by_field(field_id)
    return {
        "field_map": fmap.model_dump(),
        "geojson": fmap.to_geojson(crops),
        "summary": mem.get_field_summary(field_id),
    }


@app.get("/field-map/summary")
def get_field_summary(field_id: str = "F01"):
    mem, _, _, _ = _get_memory_and_kb()
    return mem.get_field_summary(field_id)


@app.get("/crops")
def list_crops(
    field_id: str = "F01",
    status: Optional[str] = None,
    row: Optional[str] = None,
):
    mem, _, _, _ = _get_memory_and_kb()
    crops = mem.get_all_by_field(field_id)
    if status:
        crops = [c for c in crops if c.status.value == status.lower()]
    if row:
        crops = [c for c in crops if c.row_id.upper() == row.upper()]
    return {"count": len(crops), "crops": [c.model_dump() for c in crops]}


@app.get("/crop/{crop_id}")
def get_crop_record(crop_id: str):
    mem, _, _, kb = _get_memory_and_kb()
    rec = mem.get(crop_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Crop {crop_id} not found")
    return {
        "crop": rec.model_dump(),
        "agro_context": kb.build_context_for_slm(rec),
    }


@app.get("/crop/{crop_id}/history")
def get_crop_history(crop_id: str):
    mem, _, _, _ = _get_memory_and_kb()
    rec = mem.get(crop_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Crop {crop_id} not found")
    return {
        "crop_id": rec.crop_id,
        "status": rec.status.value,
        "total_dose_ml": rec.total_dose_ml,
        "total_doses_count": rec.total_doses_count,
        "doses_applied": [d.model_dump() for d in rec.doses_applied],
    }


@app.post("/mission/plan")
def plan_mission(req: MissionPlanRequest):
    _, _, planner, _ = _get_memory_and_kb()
    queue = planner.plan_mission(strategy=req.strategy)
    return {"strategy": req.strategy, "queue": queue, "progress": planner.get_progress()}


@app.get("/mission/progress")
def get_mission_progress():
    _, _, planner, _ = _get_memory_and_kb()
    return planner.get_progress()


@app.post("/knowledge/query")
def query_pomegranate_knowledge(req: KnowledgeQueryRequest):
    mem, _, _, kb = _get_memory_and_kb()
    q_low = req.question.lower()
    matched_sections = {}

    for st_key in kb.kb.get("phenology", {}).get("growth_stages", {}):
        if st_key.replace("_", " ") in q_low or st_key in q_low:
            matched_sections[f"stage:{st_key}"] = kb.get_dosing_protocol(st_key)
    for d_key in kb.kb.get("diseases", {}):
        if d_key.replace("_", " ") in q_low or d_key in q_low:
            matched_sections[f"disease:{d_key}"] = kb.get_disease_info(d_key)
    for p_key in kb.kb.get("pests", {}):
        if p_key.replace("_", " ") in q_low or p_key in q_low:
            matched_sections[f"pest:{p_key}"] = kb.get_pest_info(p_key)
    for nut_key in kb.kb.get("nutrient_management", {}).get("deficiency_symptoms", {}):
        if nut_key in q_low:
            matched_sections[f"deficiency:{nut_key}"] = kb.get_deficiency_info(nut_key)
    if "bahar" in q_low:
        matched_sections["bahar_system"] = kb.kb.get("phenology", {}).get("bahar_system", {})
    if "variety" in q_low or "varieties" in q_low or "bhagwa" in q_low:
        matched_sections["varieties"] = kb.kb.get("meta", {}).get("top_varieties_india", [])
    if not matched_sections:
        matched_sections["robot_decision_rules"] = kb.get_all_rules()

    crop_ctx = ""
    if req.crop_id:
        rec = mem.get(req.crop_id)
        if rec:
            crop_ctx = kb.build_context_for_slm(rec)

    brain = None
    if "sim_engine" in _state:
        brain = _state["sim_engine"].brain
    elif "brain" in _state:
        brain = _state["brain"]
    else:
        brain = LLMBrainSupervisor(backend="auto", ollama_model="qwen2.5:3b")
        _state["brain"] = brain

    slm_answer = None
    if brain.backend != "rule_fallback":
        sys_p = (
            "You are the Pomegranate Agronomy Expert Brain (Punica granatum). "
            "Answer concisely and accurately using ONLY the provided knowledge context."
        )
        usr_p = f"Question: {req.question}\n"
        if crop_ctx:
            usr_p += f"Active Crop: {crop_ctx}\n"
        usr_p += f"Knowledge Context: {matched_sections}"
        slm_answer = brain._query_slm(sys_p, usr_p)

    # Readable deterministic synthesis when SLM is offline/fallback
    kb_summary = None
    if not slm_answer:
        parts = []
        for k, v in matched_sections.items():
            if isinstance(v, dict):
                if "fertilizer" in v and "dose_ml" in v:
                    parts.append(f"{k}: {v.get('dose_ml')} mL {v.get('fertilizer')} (every {v.get('frequency_days')}d). {v.get('notes', '')}")
                elif "action" in v and "treatment" in v:
                    parts.append(f"{k}: action={v.get('action')}, treatment={v.get('treatment')}")
                else:
                    parts.append(f"{k}: {str(v)[:200]}")
            elif isinstance(v, list):
                parts.append(f"{k}: {len(v)} entries")
        kb_summary = " | ".join(parts) if parts else "Matched pomegranate KB."

    return {
        "question": req.question,
        "slm_backend": brain.backend,
        "slm_answer": slm_answer or kb_summary,
        "crop_context": crop_ctx or None,
        "matched_kb": matched_sections,
    }
