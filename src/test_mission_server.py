"""
Integration Test for FastAPI Mission Control Server
===================================================
Verifies /health, /command, /inspect-and-authorize (positive + negative),
3D depth lifting, 6-DOF IK computation, and /audit-log using FastAPI TestClient.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.mission_server import app


def main():
    client = TestClient(app)

    print("=== 1. /health ===")
    r = client.get("/health")
    assert r.status_code == 200, r.text
    print(json.dumps(r.json(), indent=2))

    print("\n=== 2. /command ===")
    r = client.post("/command", json={
        "command_text": "Apply 120 mL of bio-fertilizer to the pomegranate base."
    })
    assert r.status_code == 200, r.text
    cmd_data = r.json()
    print(json.dumps(cmd_data, indent=2))
    assert cmd_data["intent"]["dose_volume_ml"] == 120.0
    assert cmd_data["intent"]["min_confidence"] >= 0.80

    print("\n=== 3. /inspect-and-authorize (POSITIVE: 110.jpg) ===")
    pos_path = Path("data/real_test_samples/pomegranate/110.jpg")
    with open(pos_path, "rb") as f:
        r = client.post(
            "/inspect-and-authorize",
            files={"file": ("110.jpg", f, "image/jpeg")},
            data={"execute_actuation": "true", "use_depth_3d": "true"},
        )
    assert r.status_code == 200, r.text
    pos = r.json()
    print(json.dumps(pos, indent=2))
    assert pos["decision"]["authorized"] is True
    assert pos["ik_joint_angles_rad"] is not None
    assert len(pos["ik_joint_angles_rad"]) == 6
    assert pos["safe_abort"] is False
    assert pos["hardware_frame"]["safety_state"] == "COMPLETED_SAFE"
    assert pos["hardware_frame"]["retract_verified"] is True
    assert len(pos["safety_gates"]) == 4

    print("\n=== 3B. FAULT INJECTION TESTS (4-GATE SAFE ABORT) ===")
    with open(pos_path, "rb") as f:
        r_tilt = client.post(
            "/inspect-and-authorize",
            files={"file": ("110.jpg", f, "image/jpeg")},
            data={"execute_actuation": "true", "imu_roll_deg": "18.5"},
        )
    assert r_tilt.json()["hardware_frame"]["safety_state"] == "SAFE_ABORT_UNSTABLE"
    assert r_tilt.json()["actuation"] is None

    with open(pos_path, "rb") as f:
        r_tol = client.post(
            "/inspect-and-authorize",
            files={"file": ("110.jpg", f, "image/jpeg")},
            data={"execute_actuation": "true", "nozzle_error_cm": "9.2"},
        )
    assert r_tol.json()["hardware_frame"]["safety_state"] == "SAFE_ABORT_OUT_OF_TOLERANCE"
    assert r_tol.json()["actuation"] is None

    with open(pos_path, "rb") as f:
        r_pump = client.post(
            "/inspect-and-authorize",
            files={"file": ("110.jpg", f, "image/jpeg")},
            data={"execute_actuation": "true", "measured_flow_ml": "10.0"},
        )
    assert r_pump.json()["hardware_frame"]["safety_state"] == "SAFE_ABORT_PUMP_FAULT"
    assert r_pump.json()["safe_abort"] is True
    print("[✓] All 3 hardware fault-injection Safe Aborts passed (IMU tilt, nozzle tolerance, pump fault)!")

    print("\n=== 4. /inspect-and-authorize (NEGATIVE: bare_soil_control.jpg) ===")
    neg_path = Path("data/real_test_samples/negative/bare_soil_control.jpg")
    with open(neg_path, "rb") as f:
        r = client.post(
            "/inspect-and-authorize",
            files={"file": ("bare_soil_control.jpg", f, "image/jpeg")},
            data={"execute_actuation": "true", "use_depth_3d": "true"},
        )
    assert r.status_code == 200, r.text
    neg = r.json()
    print(json.dumps(neg, indent=2))
    assert neg["decision"]["authorized"] is False
    assert neg["ik_joint_angles_rad"] is None

    print("\n=== 5. /audit-log ===")
    r = client.get("/audit-log?limit=3")
    assert r.status_code == 200, r.text
    print(json.dumps(r.json(), indent=2))
    print("\n[✓] ALL FASTAPI MISSION SERVER INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    main()
