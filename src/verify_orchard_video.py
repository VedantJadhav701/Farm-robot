"""
Orchard Video Validation Pipeline (System 1 + 3D Depth + Arm Strategy)
======================================================================
Runs the Dual-Brain + Low-Branch Collision Planner frame-by-frame on any
orchard smartphone video (.mp4) and exports an annotated proof video + JSON.

Usage:
  python src/verify_orchard_video.py --input path/to/orchard.mp4 --max-frames 90
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.perception.depth_localizer import MonocularDepthLocalizer
from src.control.kinematics import ArmKinematics


def main():
    parser = argparse.ArgumentParser(description="Run Dual-Brain on orchard video")
    parser.add_argument("--input", default="gemini_generated_video_9aeb227e.mp4")
    parser.add_argument("--output", default="output/orchard_video_validation.mp4")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--depth-every", type=int, default=10, help="Run depth every N frames")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Video not found: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = PomegranateVisionDetector(weights_path="models/best.pt", confidence_threshold=0.70, nms_conf=0.25)
    depth = MonocularDepthLocalizer()
    kin = ArmKinematics()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))

    last_disp = None
    last_strategy = "SCANNING"
    last_pour = [0.0, 0.0, 0.0]
    frames_done = 0
    detected_frames = 0
    s1_times = []

    while frames_done < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        vis = detector.analyze_crop_image(frame)
        s1_times.append(vis.inference_ms)
        if vis.is_pomegranate:
            detected_frames += 1

        if frames_done % args.depth_every == 0 or last_disp is None:
            last_disp, _ = depth.estimate_depth_map(frame)
            if vis.is_pomegranate:
                trunk_3d = depth.pixel_to_3d(vis.trunk_base_pixel, last_disp)
                pour_3d = kin.compute_precision_pour_target(trunk_3d, 0.125)
                obs = kin.analyze_low_branch_obstacles(vis.all_detections, last_disp, depth)
                plan = kin.plan_collision_free_approach(np.array([0.0, 0.45, -0.25]), pour_3d, obs)
                last_strategy = plan["strategy"]
                last_pour = plan["pour_target_m"]

        rgb_ann = detector.draw_detections(frame, vis)
        cv2.putText(rgb_ann, f"Strategy: {last_strategy} | Pour: {last_pour}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
        d_col = depth.colorize_depth(last_disp)
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
        "avg_system1_ms": round(sum(s1_times) / max(1, len(s1_times)), 2),
        "last_strategy": last_strategy,
    }
    Path("output/orchard_video_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
