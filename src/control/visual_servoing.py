"""
Eye-in-Hand Image-Based Visual Servoing (IBVS) Controller
=========================================================
Closed-loop visual servoing for the 6-DOF orchard arm after System 1+2
verification. Drives joint angles (yaw, shoulder, elbow, wrist_pitch)
using pixel error (ex, ey) toward the 12.5 cm basal pour ring.
"""

from typing import Tuple, Dict, Any
import numpy as np


class EyeInHandVisualServo:
    def __init__(
        self,
        gain_yaw: float = 0.22,
        gain_pitch: float = 0.18,
        tolerance_cm: float = 4.5,
    ):
        self.gain_yaw = gain_yaw
        self.gain_pitch = gain_pitch
        self.tolerance_cm = tolerance_cm

    def step(
        self,
        current_q: np.ndarray,
        target_px: Tuple[int, int],
        frame_shape: Tuple[int, int] = (480, 640),
        goal_px: Tuple[int, int] = (420, 390),
    ) -> Dict[str, Any]:
        """Compute closed-loop joint update toward visual trunk-base goal."""
        h, w = frame_shape
        ex = (target_px[0] - goal_px[0]) / max(1.0, float(w))
        ey = (target_px[1] - goal_px[1]) / max(1.0, float(h))

        # Approximate ground-plane metric error (cm) at ~1.0m arm reach
        err_cm = float(np.hypot(ex * 110.0, ey * 85.0))

        # Base pour posture + closed-loop IBVS correction
        base_q = np.array([1.45, 0.95, -0.55, 0.0, -0.45, 0.0], dtype=np.float64)
        dq_yaw = float(np.clip(-self.gain_yaw * ex, -0.25, 0.25))
        dq_pitch = float(np.clip(self.gain_pitch * ey, -0.20, 0.20))

        desired_q = base_q.copy()
        desired_q[0] = float(np.clip(base_q[0] + dq_yaw, -2.8, 2.8))
        desired_q[1] = float(np.clip(base_q[1] + dq_pitch, -1.6, 1.6))
        desired_q[4] = float(np.clip(base_q[4] - 0.5 * dq_pitch, -1.6, 1.6))

        next_q = 0.80 * np.asarray(current_q, dtype=np.float64) + 0.20 * desired_q
        return {
            "next_q": next_q,
            "error_cm": round(err_cm, 2),
            "aligned": err_cm <= self.tolerance_cm,
        }
