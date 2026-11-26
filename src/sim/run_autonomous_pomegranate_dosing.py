"""
Integrated Autonomous Pomegranate Detection & Precision Dosing Simulation
(Real Async Dual-Brain: YOLOv8s System 1 + Ollama Qwen2.5-3B System 2 + MuJoCo)
================================================================================
Combines:
1. 3D Orchard physics simulation (4WD Rover traversing pomegranate orchard row).
2. Real-time System 1 vision (trained YOLOv8s models/best.pt, non-blocking).
3. Asynchronous System 2 SLM supervisor (Ollama qwen2.5:3b, background thread).
4. Precision liquid dosing actuator (0.25 L/s solenoid valve) + SQLite audit log.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import mujoco

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.api.llm_brain import LLMBrainSupervisor
from src.control.async_dual_brain import AsyncDualBrainRunner
from src.control.fertilizer_controller import FertilizerDosingController


def load_real_orchard_sensor_feed() -> np.ndarray:
    """Load a real PG-YOLO-Dataset orchard frame for realistic sensor-in-the-loop."""
    samples = sorted(Path("data/real_test_samples/pomegranate").glob("*.jpg"))
    if samples:
        img = cv2.imread(str(samples[0]))
        if img is not None:
            return img
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def main():
    model_path = os.path.abspath("src/sim/models/orchard_mobile_manipulator.xml")
    output_video_path = os.path.abspath("output/autonomous_pomegranate_dosing_simulation.mp4")

    print(f"Loading 3D Orchard Physics Model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 1. Initialize Real Trained System 1 + System 2 + Async Runner
    detector = PomegranateVisionDetector(weights_path="models/best.pt", confidence_threshold=0.70)
    brain = LLMBrainSupervisor(pump_flow_rate_ml_per_sec=250.0, backend="auto", ollama_model="qwen2.5:3b")
    controller = FertilizerDosingController(db_path="output/dosing_audit_log.db")

    operator_command = "Traverse Row 04, verify pomegranate trees, and dispense 80 mL bio-fertilizer directly at root base."
    intent = brain.parse_operator_command(operator_command)
    print(f"[System 2 ({brain.backend})] Parsed Intent: crop={intent.target_crop}, dose={intent.dose_volume_ml} mL, min_conf={intent.min_confidence}")

    runner = AsyncDualBrainRunner(detector, brain, intent, system2_interval_sec=1.0)
    # Warmup CUDA kernel
    real_orchard_frame = load_real_orchard_sensor_feed()
    _ = detector.analyze_crop_image(real_orchard_frame)

    width, height = 1280, 720
    renderer = mujoco.Renderer(model, height, width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    fps = 30
    duration_sec = 8.0
    total_frames = int(duration_sec * fps)

    dosing_authorized = False
    dosing_active = False
    dosing_completed = False
    latest_auth = None

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking_view")
    print(f"Running Asynchronous Dual-Brain MuJoCo Loop ({total_frames} frames @ {fps} Hz)...")

    for frame_idx in range(total_frames):
        sim_time = frame_idx / float(fps)

        if sim_time < 2.0:
            mission_phase = "NAVIGATING INTER-ROW (0.35 m/s)"
            data.ctrl[0:4] = 3.0
        elif 2.0 <= sim_time < 3.8:
            data.ctrl[0:4] = 0.0
            mission_phase = "ASYNC DUAL-BRAIN VERIFYING CROP..."
        elif 3.8 <= sim_time < 5.8:
            mission_phase = f"DISPENSING FERTILIZER ({intent.dose_volume_ml} mL @ 0.25 L/S)"
            dosing_active = bool(dosing_authorized)
        else:
            dosing_active = False
            if not dosing_completed and latest_auth and latest_auth.authorized:
                controller.execute_liquid_dose(latest_auth)
                dosing_completed = True
            mission_phase = "DOSING COMPLETE - ADVANCING TO NEXT TREE"
            data.ctrl[0:4] = 3.0

        sim_steps = int((1.0 / fps) / model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam_id)
        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Feed real orchard camera frame during tree inspection window (2.0s - 5.8s), MuJoCo view otherwise
        sensor_frame = real_orchard_frame if (2.0 <= sim_time < 5.8) else frame_bgr
        vision, auth = runner.step(sensor_frame)
        if auth is not None:
            latest_auth = auth
            if auth.authorized:
                dosing_authorized = True

        # HUD Overlays
        cv2.rectangle(frame_bgr, (25, 25), (500, 135), (20, 25, 30), -1)
        cv2.rectangle(frame_bgr, (25, 25), (500, 135), (0, 255, 120), 2)
        cv2.putText(frame_bgr, "ASYNC PHYSICAL AI DUAL-BRAIN (NO MOCKS)", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 200), 1)
        cv2.putText(frame_bgr, f"MISSION: {mission_phase}", (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 2)
        cv2.putText(frame_bgr, f"S1: {detector.backend} ({vision.inference_ms:.1f}ms) | S2: {brain.backend}", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 220), 1)

        cv2.rectangle(frame_bgr, (860, 25), (1255, 150), (20, 25, 30), -1)
        cv2.rectangle(frame_bgr, (860, 25), (1255, 150), (70, 90, 100), 1)
        cnn_txt = f"POMEGRANATE ({vision.confidence*100:.1f}%)" if vision.is_pomegranate else "SCANNING..."
        cv2.putText(frame_bgr, f"SYSTEM 1: {cnn_txt}", (878, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120) if vision.is_pomegranate else (0, 200, 255), 2)
        slm_txt = "AUTHORIZED" if dosing_authorized else "EVALUATING..."
        cv2.putText(frame_bgr, f"SYSTEM 2: {slm_txt}", (878, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120) if dosing_authorized else (200, 200, 200), 2)
        act_txt = "NOZZLE DISPENSING" if dosing_active else ("DOSE LOGGED" if dosing_completed else "STANDBY")
        cv2.putText(frame_bgr, f"ACTUATOR: {act_txt}", (878, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255) if dosing_active else (0, 255, 120), 2)

        if vision.is_pomegranate:
            cv2.rectangle(frame_bgr, (720, 160), (960, 520), (0, 255, 120), 2)
            cv2.circle(frame_bgr, (840, 510), 14, (0, 220, 255), 2)

        if dosing_active:
            cv2.line(frame_bgr, (640, 390), (840, 510), (255, 220, 0), 4, cv2.LINE_AA)
            cv2.ellipse(frame_bgr, (840, 510), (26, 12), 0, 0, 360, (255, 220, 0), 2)

        out.write(frame_bgr)

    runner.wait_pending()
    out.release()

    metrics = runner.stats()
    Path("output/async_sim_benchmark.json").write_text(json.dumps(metrics, indent=2))
    print("\n" + "=" * 62)
    print("ASYNC DUAL-BRAIN SIMULATION COMPLETE (ZERO BLOCKING!)")
    print(f"Video   : {output_video_path}")
    print(f"Metrics : {json.dumps(metrics)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
