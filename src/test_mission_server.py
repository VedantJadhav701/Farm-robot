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
