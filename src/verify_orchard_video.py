"""
Stabilized Robot-Brain Orchard Video Validation Pipeline
========================================================
Uses:
  - System 1: YOLOv8s (models/best.pt)
  - TemporalTargetStabilizer: EMA/Kalman smoothing + Brake->Stabilize->Pour->Advance
  - 3D Lifting: Depth-Anything-V2-Small INT8 (computed on lock/periodically)
  - ArmKinematics: 12.5 cm pour ring + low-branch collision strategy
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.perception.depth_localizer import MonocularDepthLocalizer
from src.control.kinematics import ArmKinematics
from src.control.target_tracker import TemporalTargetStabilizer


def main():
    parser = argparse.ArgumentParser(description="Run stabilized robot-brain on orchard video")
    parser.add_argument("--input", default="data/real_test_samples/orchard_turkey_480p.mp4")
    parser.add_argument("--output", default="output/turkey_orchard_proof.mp4")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--start-sec", type=float, default=120.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--depth-every", type=int, default=12)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Video not found: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    detector = PomegranateVisionDetector(weights_path=args.weights, confidence_threshold=0.70, nms_conf=0.30)
    depth = MonocularDepthLocalizer()
    kin = ArmKinematics()
    tracker = TemporalTargetStabilizer(ema_alpha=0.20, max_jump_px=75.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))

    last_disp = None
    last_strategy = "SCANNING"
    last_pour = [0.0, 0.0, 0.0]
    frames_done = 0
    detected_frames = 0
    pour_events = 0
    was_pouring = False
    s1_times = []

    while frames_done < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        vis = detector.analyze_crop_image(frame)
        s1_times.append(vis.inference_ms)
        if vis.is_pomegranate:
            detected_frames += 1

        t_state = tracker.update(vis.is_pomegranate, vis.trunk_base_pixel)
        if t_state["is_pouring"] and not was_pouring:
            pour_events += 1
        was_pouring = t_state["is_pouring"]

        if frames_done % args.depth_every == 0 or last_disp is None:
            last_disp, _ = depth.estimate_depth_map(frame)
            if vis.is_pomegranate or t_state["target_locked"]:
                trunk_3d = depth.pixel_to_3d(t_state["stabilized_px"], last_disp)
                pour_3d = kin.compute_precision_pour_target(trunk_3d, 0.125)
                obs = kin.analyze_low_branch_obstacles(vis.all_detections, last_disp, depth)
                plan = kin.plan_collision_free_approach(np.array([0.0, 0.45, -0.25]), pour_3d, obs)
                last_strategy = plan["strategy"]
                last_pour = plan["pour_target_m"]

        # Draw detections + stabilized locked target + robot state HUD
        vis.trunk_base_pixel = t_state["stabilized_px"]
        rgb_ann = detector.draw_detections(frame, vis)

        # Robot State Machine Banner
        cv2.rectangle(rgb_ann, (8, rgb_ann.shape[0] - 85), (rgb_ann.shape[1] - 8, rgb_ann.shape[0] - 8), (18, 22, 28), -1)
        state_color = (0, 220, 255) if t_state["is_pouring"] else ((0, 165, 255) if t_state["is_braked"] else (0, 255, 120))
        cv2.putText(rgb_ann, f"ROBOT: {t_state['state']} | v={t_state['cmd_vel_mps']:.2f} m/s",
                    (16, rgb_ann.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        cv2.putText(rgb_ann, f"ARM: {last_strategy} | 12.5cm Pour: {last_pour} m",
                    (16, rgb_ann.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 230, 240), 1)

        # Draw simulated nozzle spray jet when POURING
        if t_state["is_pouring"]:
            tx, ty = t_state["stabilized_px"]
            nozzle = (rgb_ann.shape[1] // 2, rgb_ann.shape[0] - 90)
            cv2.line(rgb_ann, nozzle, (tx, ty), (255, 220, 0), 3, cv2.LINE_AA)
            cv2.circle(rgb_ann, (tx, ty), 18, (255, 220, 0), 2)

        d_col = depth.colorize_depth(last_disp)
        cv2.drawMarker(d_col, t_state["stabilized_px"], (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        panel = np.hstack([cv2.resize(rgb_ann, (640, 360)), cv2.resize(d_col, (640, 360))])
        writer.write(panel)
        frames_done += 1

    cap.release()
    writer.release()

    summary = {
        "input_video": str(in_path),
        "output_video": str(out_path),
        "frames_processed": frames_done,
        "pomegranate_detected_frames": detected_frames,
        "autonomous_pour_events": pour_events,
        "avg_system1_ms": round(sum(s1_times) / max(1, len(s1_times)), 2),
        "last_strategy": last_strategy,
    }
    Path("output/orchard_video_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
