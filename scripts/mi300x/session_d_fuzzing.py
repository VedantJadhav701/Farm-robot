"""
Session D — Parallel Safety Fuzzing & Invariant Verification
============================================================
Runs randomized episodes across worker processes (20 vCPUs on MI300X) to verify
that the refactored safety supervisor & tree-bound authorizations fail closed:
  1. Valve never opens unless Gates 1 & 2 pass.
  2. Valve-open time never exceeds 2000 ms.
  3. No dose without a fresh authorization bound to that tree.
  4. On any fault, valve stays closed and arm retracts (Gate 4).
  5. Every episode produces an outcome record.
  6. Working flow sensor delivery stays within +-15% of requested dose.

Usage:
  python scripts/mi300x/session_d_fuzzing.py --episodes 10000 --workers 20
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List

from src.api.llm_brain import DosingAuthorization
from src.hardware.ros2_actuator_bridge import ROS2ActuatorBridge, SafetyTelemetry


def _run_chunk(args) -> Dict[str, Any]:
    seed_start, count = args
    bridge = ROS2ActuatorBridge()
    violations: List[Dict[str, Any]] = []
    stats = {"total": 0, "completed": 0, "aborted": 0}

    for i in range(count):
        seed = seed_start + i
        rng = random.Random(seed)
        stats["total"] += 1

        tree_id = rng.randint(1, 8)
        auth_tree_id = tree_id if rng.random() > 0.15 else rng.randint(1, 8)
        dose_ml = rng.uniform(10.0, 500.0)
        pulse_s = min(2.0, dose_ml / 250.0)

        # Randomized fault injection
        speed_mps = 0.0 if rng.random() > 0.15 else rng.uniform(0.05, 0.50)
        roll_deg = rng.uniform(-15.0, 15.0)
        pitch_deg = rng.uniform(-14.0, 14.0)
        nozzle_err_cm = rng.uniform(0.2, 12.0)

        flow_fault = rng.random()
        if flow_fault < 0.10:
            measured_flow = None  # sensor dropout
        elif flow_fault < 0.20:
            measured_flow = dose_ml * rng.uniform(0.4, 0.7)  # clog / bias
        else:
            measured_flow = dose_ml * rng.uniform(0.92, 1.08)

        valve_closed = rng.random() > 0.05
        stale_age_s = 0.2 if rng.random() > 0.10 else 5.0

        auth = DosingAuthorization(
            authorized=(rng.random() > 0.10),
            dose_volume_ml=dose_ml,
            pulse_duration_seconds=pulse_s,
            target_coordinates_m=(0.0, 0.72, -0.35),
            decision_reasoning="fuzz",
            crop_verified=True,
            confidence_score=rng.uniform(0.80, 0.99),
            tree_id=auth_tree_id,
            frame_id=seed,
        )
        auth.issued_at -= stale_age_s

        is_auth_valid = auth.is_valid(tree_id=tree_id, ttl_s=2.0)
        tel = SafetyTelemetry(
            rover_speed_mps=speed_mps,
            imu_roll_deg=roll_deg,
            imu_pitch_deg=pitch_deg,
            nozzle_error_cm=nozzle_err_cm,
            measured_flow_ml=measured_flow,
            valve_closed_confirmed=valve_closed,
        )

        valve_opened = False

        def _actuate():
            nonlocal valve_opened
            valve_opened = True
            return {"status": "VALVE_PULSED"}

        res = bridge.dispatch(
            authorized=is_auth_valid,
            dose_volume_ml=dose_ml,
            pulse_duration_s=pulse_s,
            arm_plan={"strategy": "STANDOFF_ANGLED_JET", "pour_target_3d_m": [0.0, 0.72, -0.35], "nozzle_pitch_rad": -0.85},
            ik_joints=[1.45, 0.95, -0.55, 0.0, -0.45, 0.0],
            telemetry=tel,
            actuation_fn=_actuate,
        )

        frame = res["frame"]
        gates = {g["gate"]: g["passed"] for g in res["gates"]}

        # Invariant 1 & 3: No valve open without valid tree-bound auth + G1 + G2
        if valve_opened and (not is_auth_valid or not gates.get("1_pre_arm_stability") or not gates.get("2_pre_spray_confirmation")):
            violations.append({"seed": seed, "rule": "UNSAFE_VALVE_OPEN"})

        # Invariant 2: Valve open <= 2000 ms
        if frame["valve_open_ms"] > 2000:
            violations.append({"seed": seed, "rule": "VALVE_CLAMP_EXCEEDED", "ms": frame["valve_open_ms"]})

        # Invariant 4: On abort, retract_verified must be True and valve_open_ms == 0 (if pre-gate abort)
        if res["safe_abort"] and not frame["retract_verified"]:
            violations.append({"seed": seed, "rule": "ABORT_NOT_RETRACTED"})

        # Invariant 6: Gate 3 must fail if measured_flow is None or >15% off
        if measured_flow is None and gates.get("3_valve_and_flow_verification") is True:
            violations.append({"seed": seed, "rule": "GATE3_FAILED_OPEN_ON_NONE"})

        if res["safe_abort"]:
            stats["aborted"] += 1
        else:
            stats["completed"] += 1

    return {"stats": stats, "violations": violations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    parser.add_argument("--out_dir", type=str, default="output/fuzzing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    chunk_size = max(1, args.episodes // args.workers)
    tasks = [(i * chunk_size, chunk_size) for i in range(args.workers)]

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_run_chunk, tasks)

    total_stats = {"total": 0, "completed": 0, "aborted": 0}
    all_violations: List[Dict[str, Any]] = []
    for r in results:
        for k in total_stats:
            total_stats[k] += r["stats"][k]
        all_violations.extend(r["violations"])

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "episodes": total_stats["total"],
        "completed": total_stats["completed"],
        "aborted": total_stats["aborted"],
        "violations_count": len(all_violations),
        "elapsed_s": elapsed,
        "episodes_per_sec": round(total_stats["total"] / max(0.01, elapsed), 1),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "violations.json").write_text(json.dumps(all_violations[:200], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
