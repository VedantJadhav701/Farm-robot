"""
Real-World Dual-Brain Verification & RTX 3050 Edge Benchmark
=============================================================
Tests the trained System 1 (YOLOv8s best.pt) + System 2 (Ollama qwen2.5:3b)
end-to-end on real pomegranate orchard images and negative controls.

Outputs:
  - output/real_proof_*.jpg (annotated bounding boxes + pour targets + HUD)
  - output/local_dual_brain_verification_report.json
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np

from src.perception.pomegranate_detector import PomegranateVisionDetector
from src.api.llm_brain import LLMBrainSupervisor
from src.control.fertilizer_controller import FertilizerDosingController


def create_bare_soil_negative(path: Path):
    """Create a realistic textured bare-soil / dry-grass negative control."""
    rng = np.random.default_rng(42)
    img = np.full((720, 1280, 3), (42, 54, 70), dtype=np.uint8)
    noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main():
    print("=" * 70)
    print("  REAL-WORLD DUAL-BRAIN VERIFICATION (SYSTEM 1 YOLO + SYSTEM 2 SLM)")
    print("=" * 70)

    # 1. Initialize System 1 (trained YOLOv8s) & System 2 (local Ollama qwen2.5:3b)
    detector = PomegranateVisionDetector(
        weights_path="models/best.pt",
        confidence_threshold=0.70,
        nms_conf=0.25,
    )
    brain = LLMBrainSupervisor(
        pump_flow_rate_ml_per_sec=250.0,
        backend="auto",
        ollama_model="qwen2.5:3b",
    )
    controller = FertilizerDosingController(db_path="output/dosing_audit_log.db")

    print(f"[System 1] Backend: {detector.backend} | Device: {detector.device}")
    print(f"[System 2] Backend: {brain.backend} | Model: {brain.ollama_model}")

    # 2. Ensure negative control exists
    neg_path = Path("data/real_test_samples/negative/bare_soil_control.jpg")
    if not neg_path.exists():
        create_bare_soil_negative(neg_path)

    # Warmup System 1 CUDA kernel on dummy frame so steady-state latency is measured
    _ = detector.analyze_crop_image(np.zeros((640, 640, 3), dtype=np.uint8))

    # 3. Parse real operator command via System 2 SLM
    cmd = "Apply 80 mL of bio-fertilizer to the pomegranate base."
    t_slm0 = time.perf_counter()
    intent = brain.parse_operator_command(cmd)
    intent_ms = (time.perf_counter() - t_slm0) * 1000
    print(f"\n[Operator] '{cmd}'")
    print(f"[System 2 Intent] ({intent_ms:.0f} ms) -> crop={intent.target_crop}, "
          f"dose={intent.dose_volume_ml} mL, min_conf={intent.min_confidence}")

    # 4. Run on real pomegranate orchard images
    pos_imgs = sorted(Path("data/real_test_samples/pomegranate").glob("*.*"))
    neg_imgs = sorted(Path("data/real_test_samples/negative").glob("*.*"))

    report = {
        "system1_backend": detector.backend,
        "system1_device": str(detector.device),
        "system2_backend": brain.backend,
        "system2_model": brain.ollama_model,
        "intent_parse_ms": round(intent_ms, 1),
        "positive_samples": [],
        "negative_samples": [],
    }

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("\n--- POSITIVE SAMPLES (REAL POMEGRANATE ORCHARD IMAGES) ---")
    for idx, img_path in enumerate(pos_imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        vision = detector.analyze_crop_image(frame)
        t0 = time.perf_counter()
        decision = brain.evaluate_dosing_decision(intent, vision)
        dec_ms = (time.perf_counter() - t0) * 1000

        if decision.authorized:
            controller.execute_liquid_dose(decision)

        # Save annotated proof for first 3 images
        if idx < 3:
            annotated = detector.draw_detections(frame, vision)
            status_txt = "AUTHORIZED" if decision.authorized else "ABORTED"
            color = (0, 255, 0) if decision.authorized else (0, 0, 255)
            cv2.putText(annotated, f"System 2 ({brain.backend}): {status_txt}",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(str(out_dir / f"real_proof_{idx+1}_{img_path.stem}.jpg"), annotated)

        entry = {
            "image": img_path.name,
            "is_pomegranate": vision.is_pomegranate,
            "confidence": vision.confidence,
            "num_detections": len(vision.all_detections),
            "system1_ms": vision.inference_ms,
            "system2_ms": round(dec_ms, 1),
            "authorized": decision.authorized,
            "dose_ml": decision.dose_volume_ml,
            "reasoning": decision.decision_reasoning,
        }
        report["positive_samples"].append(entry)
        print(f"  [{img_path.name}] dets={len(vision.all_detections)} | conf={vision.confidence:.1%} | "
              f"S1={vision.inference_ms:.1f}ms | S2={dec_ms:.0f}ms | auth={decision.authorized}")

    print("\n--- NEGATIVE CONTROLS (NON-POMEGRANATE) ---")
    for img_path in neg_imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        vision = detector.analyze_crop_image(frame)
        t0 = time.perf_counter()
        decision = brain.evaluate_dosing_decision(intent, vision)
        dec_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "image": img_path.name,
            "is_pomegranate": vision.is_pomegranate,
            "confidence": vision.confidence,
            "num_detections": len(vision.all_detections),
            "system1_ms": vision.inference_ms,
            "system2_ms": round(dec_ms, 1),
            "authorized": decision.authorized,
            "reasoning": decision.decision_reasoning,
        }
        report["negative_samples"].append(entry)
        print(f"  [{img_path.name}] dets={len(vision.all_detections)} | conf={vision.confidence:.1%} | "
              f"auth={decision.authorized} | {decision.decision_reasoning[:70]}")

    # Summary stats
    s1_times = [e["system1_ms"] for e in report["positive_samples"] if e["system1_ms"] > 0]
    if s1_times:
        avg_s1 = sum(s1_times) / len(s1_times)
        report["avg_system1_ms"] = round(avg_s1, 1)
        report["avg_system1_fps"] = round(1000.0 / avg_s1, 1)
        print(f"\n[Summary] Avg System 1 RTX 3050 Latency: {avg_s1:.1f} ms ({1000/avg_s1:.1f} FPS)")

    rep_path = out_dir / "local_dual_brain_verification_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[✓] Saved verification report → {rep_path}")


if __name__ == "__main__":
    main()
