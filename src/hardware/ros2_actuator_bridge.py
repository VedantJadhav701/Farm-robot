"""
ROS2 / Edge Hardware Actuator Bridge (Jetson Orin / Rover Rig)
================================================================
Translates Dual-Brain decisions (System 1 + 3D Depth + ArmPlan + System 2)
into deterministic hardware control frames for:
  - /cmd_vel          (rover chassis velocity)
  - /arm_ik_target    (6-DOF joint angles + approach strategy)
  - /valve_pulse      (solenoid GPIO / CAN-bus pulse duration in ms)

Works in two modes:
  - 'ros2': Publishes native ROS2 rclpy topics when running on Jetson/Ubuntu ROS2.
  - 'dry_run_sim': Emits hardware-contract JSON frames locally on Windows for HIL testing.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


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
        except ImportError:
            self.mode = "dry_run_sim"
            logger.info("[Bridge] rclpy not found; running in local hardware-contract dry_run_sim mode.")

    def dispatch(
        self,
        authorized: bool,
        dose_volume_ml: float,
        pulse_duration_s: float,
        arm_plan: Optional[dict],
        ik_joints: Optional[List[float]],
    ) -> HardwareActuationFrame:
        if not authorized or arm_plan is None or ik_joints is None:
            frame = HardwareActuationFrame(
                timestamp_unix=round(time.time(), 3),
                rover_stop=False,
                cmd_vel_linear_mps=0.35,
                arm_strategy="STOWED",
                joint_angles_rad=[0.0, -0.6, 1.2, 0.0, -0.6, 0.0],
                nozzle_pitch_rad=-1.57,
                pour_target_3d_m=[0.0, 0.0, 0.0],
                valve_open_ms=0,
                dose_volume_ml=0.0,
                safety_authorized=False,
            )
        else:
            frame = HardwareActuationFrame(
                timestamp_unix=round(time.time(), 3),
                rover_stop=True,
                cmd_vel_linear_mps=0.0,
                arm_strategy=arm_plan["strategy"],
                joint_angles_rad=ik_joints,
                nozzle_pitch_rad=float(arm_plan["nozzle_pitch_rad"]),
                pour_target_3d_m=list(arm_plan["pour_target_3d_m"]),
                valve_open_ms=int(round(pulse_duration_s * 1000)),
                dose_volume_ml=float(dose_volume_ml),
                safety_authorized=True,
            )

        if self.mode == "ros2" and self._ros_node is not None:
            from std_msgs.msg import String
            msg = String()
            msg.data = json.dumps(asdict(frame))
            self._pub.publish(msg)

        return frame
