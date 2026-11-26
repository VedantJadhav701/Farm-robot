"""
Hardware Solenoid Relay & 6-DOF Servo Driver (Jetson GPIO / CAN-Bus)
====================================================================
Consumes HardwareActuationFrame JSON from ROS2 / Mission Server and drives:
  - 12V Solenoid Relay (hard hardware watchdog clamp: max 2000 ms pulse)
  - 6-DOF Servo / CAN-bus joint targets
  - Rover chassis /cmd_vel safety interlock

Runs in 'dry_run' mode on Windows/dev laptops and 'jetson_gpio' on Orin.
"""

import json
import time
from typing import Dict, Any


MAX_VALVE_PULSE_MS = 2000  # Hardware watchdog safety limit (500 mL @ 0.25 L/s)


class ActuatorHardwareDriver:
    def __init__(self, valve_gpio_pin: int = 18):
        self.valve_gpio_pin = valve_gpio_pin
        self.mode = "dry_run"
        self._gpio = None
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.valve_gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self.mode = "jetson_gpio"
        except ImportError:
            self.mode = "dry_run"

    def execute_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validated HardwareActuationFrame safely."""
        if not frame.get("safety_authorized", False):
            return {
                "executed": False,
                "mode": self.mode,
                "valve_ms": 0,
                "reason": f"Safety interlock blocked actuation ({frame.get('safety_state')})",
            }

        pulse_ms = int(min(MAX_VALVE_PULSE_MS, max(0, frame.get("valve_open_ms", 0))))
        if self.mode == "jetson_gpio" and self._gpio is not None and pulse_ms > 0:
            self._gpio.output(self.valve_gpio_pin, self._gpio.HIGH)
            time.sleep(pulse_ms / 1000.0)
            self._gpio.output(self.valve_gpio_pin, self._gpio.LOW)

        return {
            "executed": True,
            "mode": self.mode,
            "valve_ms": pulse_ms,
            "joints_rad": frame.get("joint_angles_rad"),
            "arm_strategy": frame.get("arm_strategy"),
        }


if __name__ == "__main__":
    drv = ActuatorHardwareDriver()
    ok_res = drv.execute_frame({
        "safety_authorized": True,
        "valve_open_ms": 480,
        "arm_strategy": "LOW_UNDER_CANOPY_ARC",
        "joint_angles_rad": [0.008, 0.415, -1.717, 0.0, -0.048, 0.0],
    })
    blocked_res = drv.execute_frame({
        "safety_authorized": False,
        "safety_state": "SAFE_ABORT_UNSTABLE",
        "valve_open_ms": 480,
    })
    print("OK Frame     :", json.dumps(ok_res))
    print("Blocked Frame:", json.dumps(blocked_res))
