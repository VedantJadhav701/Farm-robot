"""
Temporal Target Tracker & Robot Hysteresis State Machine
========================================================
Implements the stabilization pattern used in NVIDIA Isaac ROS / ByteTrack
agricultural robotics pipelines:
  - 2D Kalman-style EMA smoothing with jump rejection (eliminates crosshair jitter)
  - Temporal hysteresis state machine:
      SCANNING_ROW -> BRAKING_STABILIZE -> POURING -> RETRACT_ADVANCE
  - Freezes the 3D/2D pour target during BRAKING_STABILIZE & POURING so
    the nozzle never chases 1-frame detection flickers.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np


class RobotVideoState(Enum):
    SCANNING_ROW = "SCANNING_ROW (0.35 m/s)"
    BRAKING_STABILIZE = "BRAKING & LOCKING TARGET (0.0 m/s)"
    POURING = "DISPENSING 80 mL @ 12.5cm RING"
    RETRACT_ADVANCE = "RETRACT ARM & ADVANCE (0.35 m/s)"


class TemporalTargetStabilizer:
    def __init__(self, ema_alpha: float = 0.22, max_jump_px: float = 80.0):
        self.ema_alpha = ema_alpha
        self.max_jump_px = max_jump_px
        self._smoothed_px: Optional[np.ndarray] = None
        self._locked_px: Optional[Tuple[int, int]] = None
        self._miss_streak = 0

        self.state = RobotVideoState.SCANNING_ROW
        self._hit_streak = 0
        self._state_frames = 0
        self._cooldown_frames = 0

    def reset_lock(self):
        self._locked_px = None

    def update(self, detected: bool, raw_px: Tuple[int, int]) -> dict:
        self._state_frames += 1
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1

        # ── 1. Temporal EMA smoothing with outlier-jump damping ──
        raw_arr = np.array(raw_px, dtype=np.float32)
        if detected:
            self._miss_streak = 0
            if self._smoothed_px is None:
                self._smoothed_px = raw_arr
            else:
                delta = raw_arr - self._smoothed_px
                dist = float(np.linalg.norm(delta))
                if dist > self.max_jump_px:
                    # Clamp sudden single-frame jumps instead of snapping wildly
                    delta = delta * (self.max_jump_px / dist)
                self._smoothed_px = self._smoothed_px + self.ema_alpha * delta
        else:
            self._miss_streak += 1
            if self._miss_streak > 15:
                self._smoothed_px = None

        # ── 2. Robot Hysteresis State Machine ─────────────────────
        if self.state == RobotVideoState.SCANNING_ROW:
            if detected and self._cooldown_frames == 0:
                self._hit_streak += 1
            else:
                self._hit_streak = 0

            if self._hit_streak >= 6 and self._smoothed_px is not None:
                self.state = RobotVideoState.BRAKING_STABILIZE
                self._state_frames = 0
                self._hit_streak = 0
                self._locked_px = (int(self._smoothed_px[0]), int(self._smoothed_px[1]))

        elif self.state == RobotVideoState.BRAKING_STABILIZE:
            # Refine lock gently for 12 frames while rover brakes to 0 m/s
            if self._smoothed_px is not None and self._state_frames <= 10:
                self._locked_px = (int(self._smoothed_px[0]), int(self._smoothed_px[1]))
            if self._state_frames >= 15:
                self.state = RobotVideoState.POURING
                self._state_frames = 0

        elif self.state == RobotVideoState.POURING:
            # Pour target is FROZEN during active liquid dispensing (~0.4s = 12 frames)
            if self._state_frames >= 14:
                self.state = RobotVideoState.RETRACT_ADVANCE
                self._state_frames = 0
                self._cooldown_frames = 35  # Skip re-triggering on same tree

        elif self.state == RobotVideoState.RETRACT_ADVANCE:
            if self._state_frames >= 12:
                self.state = RobotVideoState.SCANNING_ROW
                self._state_frames = 0
                self._locked_px = None

        active_px = (
            self._locked_px
            if self._locked_px is not None
            else ((int(self._smoothed_px[0]), int(self._smoothed_px[1])) if self._smoothed_px is not None else raw_px)
        )
        cmd_vel = 0.0 if self.state in (RobotVideoState.BRAKING_STABILIZE, RobotVideoState.POURING) else 0.35

        return {
            "state": self.state.value,
            "is_pouring": self.state == RobotVideoState.POURING,
            "is_braked": cmd_vel == 0.0,
            "cmd_vel_mps": cmd_vel,
            "stabilized_px": active_px,
            "target_locked": self._locked_px is not None,
        }
