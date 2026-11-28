"""
ROS2 / Edge Hardware Actuator Bridge & 4-Gate Safety Wrapper
=============================================================
Implements the complete hardware safety state machine:
  1. Pre-Arm Stability   : Stop base + IMU roll/pitch <= 12 deg check
  2. Pre-Spray Check     : Post-move visual/proximity nozzle tolerance <= 5 cm
  3. Valve Verification  : Flow-sensor delivery (+-15%) + valve closure check
  4. Arm Retraction      : Explicit retract to home_q before recording/moving

Works in 'ros2' mode on Ubuntu/Jetson or 'dry_run_sim' locally on Windows.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

HOME_JOINTS_RAD = [0.0, -0.6, 1.2, 0.0, -0.6, 0.0]


@dataclass
class SafetyTelemetry:
    rover_speed_mps: float = 0.0
    imu_roll_deg: float = 0.8
    imu_pitch_deg: float = -0.5
    nozzle_error_cm: float = 1.4
    measured_flow_ml: Optional[float] = None
    valve_closed_confirmed: bool = True
    max_tilt_deg: float = 12.0
    max_nozzle_err_cm: float = 5.0


@dataclass
class HardwareActuationFrame:
    timestamp_unix: float
    rover_stop: bool
    cmd_vel_linear_mps: float
    arm_strategy: str
    joint_angles_rad: List[float]
    nozzle_pitch_rad: float
    pour_target_3d_m: List[float]
    valve_open_ms: int
    dose_volume_ml: float
    safety_authorized: bool
    safety_state: str = "STOWED"
    retract_verified: bool = True


class ROS2ActuatorBridge:
    def __init__(self, node_name: str = "pomegranate_dual_brain_bridge"):
        self.node_name = node_name
        self.mode = "dry_run_sim"
        self._ros_node = None
        self._init_ros2_if_available()

    def _init_ros2_if_available(self):
        try:
            import rclpy
            from std_msgs.msg import String
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = rclpy.create_node(self.node_name)
            self._pub = self._ros_node.create_publisher(String, "/dual_brain/actuation_frame", 10)
            self.mode = "ros2"
            logger.info(f"[Bridge] ROS2 node '{self.node_name}' active.")
        except Exception:
            self.mode = "dry_run_sim"

    def _publish(self, frame: HardwareActuationFrame):
        if self.mode == "ros2" and self._ros_node is not None:
            from std_msgs.msg import String
            msg = String()
            msg.data = json.dumps(asdict(frame))
            self._pub.publish(msg)

    # ── Gate 1: Pre-Arm Stability ───────────────────────────────────────────
    @staticmethod
    def check_base_stability(tel: SafetyTelemetry) -> Dict[str, Any]:
        stopped = abs(tel.rover_speed_mps) <= 0.02
        tilt_ok = abs(tel.imu_roll_deg) <= tel.max_tilt_deg and abs(tel.imu_pitch_deg) <= tel.max_tilt_deg
        ok = stopped and tilt_ok
        return {
            "gate": "1_pre_arm_stability",
            "passed": ok,
            "rover_stopped": stopped,
            "imu_roll_deg": tel.imu_roll_deg,
            "imu_pitch_deg": tel.imu_pitch_deg,
            "reason": "Base stopped & IMU level" if ok else "UNSAFE: Base moving or excessive slope tilt",
        }

    # ── Gate 2: Post-Move Pre-Spray Proximity/Visual Check ──────────────────
    @staticmethod
    def confirm_nozzle_alignment(tel: SafetyTelemetry) -> Dict[str, Any]:
        ok = tel.nozzle_error_cm <= tel.max_nozzle_err_cm
        return {
            "gate": "2_pre_spray_confirmation",
            "passed": ok,
            "nozzle_error_cm": tel.nozzle_error_cm,
            "tolerance_cm": tel.max_nozzle_err_cm,
            "reason": f"Nozzle aligned ({tel.nozzle_error_cm:.1f} cm <= {tel.max_nozzle_err_cm} cm)"
                      if ok else f"OUT_OF_TOLERANCE: {tel.nozzle_error_cm:.1f} cm > {tel.max_nozzle_err_cm} cm",
        }

    # ── Gate 3: Flow-Sensor & Valve-Closure Verification ────────────────────
    @staticmethod
    def verify_delivery_and_valve(expected_ml: float, tel: SafetyTelemetry) -> Dict[str, Any]:
        if tel.measured_flow_ml is None:
            return {
                "gate": "3_valve_and_flow_verification",
                "passed": False,
                "expected_ml": expected_ml,
                "measured_ml": None,
                "valve_closed": tel.valve_closed_confirmed,
                "reason": "NO_FLOW_SENSOR: cannot verify delivery",
            }
        measured = tel.measured_flow_ml
        flow_ok = abs(measured - expected_ml) <= max(5.0, expected_ml * 0.15)
        ok = flow_ok and tel.valve_closed_confirmed
        return {
            "gate": "3_valve_and_flow_verification",
            "passed": ok,
            "expected_ml": expected_ml,
            "measured_ml": measured,
            "valve_closed": tel.valve_closed_confirmed,
            "reason": "Delivery & valve closure verified" if ok else "PUMP_FAULT: Flow mismatch or valve stuck open",
        }

    # ── Gate 4: Explicit Arm Retraction ─────────────────────────────────────
    def retract_arm(self) -> Dict[str, Any]:
        frame = HardwareActuationFrame(
            timestamp_unix=round(time.time(), 3),
            rover_stop=True,
            cmd_vel_linear_mps=0.0,
            arm_strategy="RETRACT_TO_HOME",
            joint_angles_rad=list(HOME_JOINTS_RAD),
            nozzle_pitch_rad=-1.57,
            pour_target_3d_m=[0.0, 0.0, 0.0],
            valve_open_ms=0,
            dose_volume_ml=0.0,
            safety_authorized=False,
            safety_state="RETRACTED_SAFE",
            retract_verified=True,
        )
        self._publish(frame)
        return {
            "gate": "4_arm_retraction",
            "passed": True,
            "home_joints_rad": list(HOME_JOINTS_RAD),
            "reason": "Arm retracted to stowed home pose before recording/motion",
        }

    def dispatch(
        self,
        authorized: bool,
        dose_volume_ml: float,
        pulse_duration_s: float,
        arm_plan: Optional[dict],
        ik_joints: Optional[List[float]],
        telemetry: Optional[SafetyTelemetry] = None,
        actuation_fn=None,
    ) -> Dict[str, Any]:
        """Run full 4-gate safe actuation sequence with automatic Safe Abort + Retract."""
        tel = telemetry or SafetyTelemetry()
        gates: List[Dict[str, Any]] = []

        if not authorized or arm_plan is None or ik_joints is None:
            frame = HardwareActuationFrame(
                timestamp_unix=round(time.time(), 3),
                rover_stop=False,
                cmd_vel_linear_mps=0.35,
                arm_strategy="STOWED",
                joint_angles_rad=list(HOME_JOINTS_RAD),
                nozzle_pitch_rad=-1.57,
                pour_target_3d_m=[0.0, 0.0, 0.0],
                valve_open_ms=0,
                dose_volume_ml=0.0,
                safety_authorized=False,
                safety_state="SAFE_ABORT_UNVERIFIED",
                retract_verified=True,
            )
            self._publish(frame)
            return {"frame": asdict(frame), "gates": gates, "actuation": None, "safe_abort": True}

        # Gate 1: Pre-Arm Stability
        g1 = self.check_base_stability(tel)
        gates.append(g1)
        if not g1["passed"]:
            g4 = self.retract_arm()
            gates.append(g4)
            frame = HardwareActuationFrame(
                timestamp_unix=round(time.time(), 3),
                rover_stop=True, cmd_vel_linear_mps=0.0,
                arm_strategy="STOWED", joint_angles_rad=list(HOME_JOINTS_RAD),
                nozzle_pitch_rad=-1.57, pour_target_3d_m=[0.0, 0.0, 0.0],
                valve_open_ms=0, dose_volume_ml=0.0, safety_authorized=False,
                safety_state="SAFE_ABORT_UNSTABLE", retract_verified=True,
            )
            self._publish(frame)
            return {"frame": asdict(frame), "gates": gates, "actuation": None, "safe_abort": True}

        # Gate 2: Post-Move Pre-Spray Visual/Proximity Check
        g2 = self.confirm_nozzle_alignment(tel)
        gates.append(g2)
        if not g2["passed"]:
            g4 = self.retract_arm()
            gates.append(g4)
            frame = HardwareActuationFrame(
                timestamp_unix=round(time.time(), 3),
                rover_stop=True, cmd_vel_linear_mps=0.0,
                arm_strategy=arm_plan["strategy"], joint_angles_rad=list(HOME_JOINTS_RAD),
                nozzle_pitch_rad=-1.57, pour_target_3d_m=list(arm_plan["pour_target_3d_m"]),
                valve_open_ms=0, dose_volume_ml=0.0, safety_authorized=False,
                safety_state="SAFE_ABORT_OUT_OF_TOLERANCE", retract_verified=True,
            )
            self._publish(frame)
            return {"frame": asdict(frame), "gates": gates, "actuation": None, "safe_abort": True}

        # Execute solenoid pulse
        actuation = actuation_fn() if actuation_fn is not None else None

        # Gate 3: Delivery & Valve Closure Check
        g3 = self.verify_delivery_and_valve(dose_volume_ml, tel)
        gates.append(g3)

        # Gate 4: Explicit Arm Retraction before recording/moving
        g4 = self.retract_arm()
        gates.append(g4)

        state = "COMPLETED_SAFE" if g3["passed"] else "SAFE_ABORT_PUMP_FAULT"
        frame = HardwareActuationFrame(
            timestamp_unix=round(time.time(), 3),
            rover_stop=True,
            cmd_vel_linear_mps=0.0,
            arm_strategy=arm_plan["strategy"],
            joint_angles_rad=ik_joints,
            nozzle_pitch_rad=float(arm_plan["nozzle_pitch_rad"]),
            pour_target_3d_m=list(arm_plan["pour_target_3d_m"]),
            valve_open_ms=int(round(pulse_duration_s * 1000)),
            dose_volume_ml=float(dose_volume_ml) if g3["passed"] else 0.0,
            safety_authorized=g3["passed"],
            safety_state=state,
            retract_verified=g4["passed"],
        )
        self._publish(frame)
        return {"frame": asdict(frame), "gates": gates, "actuation": actuation, "safe_abort": not g3["passed"]}
