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
from typing import List, Optional, Tuple
import numpy as np


class RobotVideoState(Enum):
    SCANNING_ROW = "SCANNING_ROW (0.35 m/s)"
    BRAKING_STABILIZE = "BRAKING & LOCKING TARGET (0.0 m/s)"
    POURING = "DISPENSING 80 mL @ 12.5cm RING"
    RETRACT_ADVANCE = "RETRACT ARM & ADVANCE (0.35 m/s)"


class TemporalTargetStabilizer:
    def __init__(self, ema_alpha: float = 0.28, max_jump_px: float = 115.0):
        self.ema_alpha = ema_alpha
        self.max_jump_px = max_jump_px
        self._smoothed_px: Optional[np.ndarray] = None
        self._vel_px: np.ndarray = np.zeros(2, dtype=np.float32)
        self._locked_px: Optional[Tuple[int, int]] = None
        self._miss_streak = 0
        self._jump_candidate: Optional[np.ndarray] = None
        self._jump_streak = 0
        self._tree_verified = False

        self.state = RobotVideoState.SCANNING_ROW
        self._hit_streak = 0
        self._state_frames = 0
        self._cooldown_frames = 0

    def reset_lock(self):
        self._locked_px = None
        self._tree_verified = False

    def _associate_trunk(
        self,
        raw_px: Tuple[int, int],
        trunk_boxes: Optional[List[Tuple[int, int, int, int, float]]],
    ) -> Tuple[np.ndarray, bool]:
        """Match the current tracked trunk by nearest predicted position (prevents cross-alley jumps)."""
        if self._smoothed_px is None or not trunk_boxes:
            return np.array(raw_px, dtype=np.float32), True

        pred = self._smoothed_px + self._vel_px
        best_pt = None
        best_dist = float("inf")
        for x1, y1, x2, y2, conf in trunk_boxes:
            if conf < 0.22:
                continue
            pt = np.array([(x1 + x2) * 0.5, float(y2)], dtype=np.float32)
            d = float(np.linalg.norm(pt - pred))
            if d < best_dist:
                best_dist = d
                best_pt = pt

        if best_pt is not None and best_dist <= self.max_jump_px:
            return best_pt, True
        return np.array(raw_px, dtype=np.float32), False

    def update(
        self,
        detected: bool,
        raw_px: Tuple[int, int],
        trunk_boxes: Optional[List[Tuple[int, int, int, int, float]]] = None,
    ) -> dict:
        self._state_frames += 1
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1

        # Latch crop identity while tracking the same physical tree
        if detected:
            self._tree_verified = True
        eff_detected = detected or (self._tree_verified and self._smoothed_px is not None and self._miss_streak <= 6)

        # ── 1. Kinematic nearest-trunk tracking + velocity prediction ──
        assoc_arr, matched_existing = self._associate_trunk(raw_px, trunk_boxes)

        if self._smoothed_px is None:
            if eff_detected or trunk_boxes:
                self._smoothed_px = assoc_arr
                self._vel_px = np.zeros(2, dtype=np.float32)
                self._miss_streak = 0
        elif matched_existing and float(np.linalg.norm(assoc_arr - self._smoothed_px)) <= self.max_jump_px:
            # Smoothly track the SAME physical trunk + update pixel velocity
            prev = self._smoothed_px.copy()
            new_pos = self._smoothed_px + self.ema_alpha * (assoc_arr - self._smoothed_px)
            inst_vel = np.clip(new_pos - prev, -18.0, 18.0)
            self._vel_px = 0.65 * self._vel_px + 0.35 * inst_vel
            self._smoothed_px = new_pos
            self._miss_streak = 0
            self._jump_candidate = None
            self._jump_streak = 0
        else:
            # Associated trunk flickered out (e.g. for 1-6 frames): coast smoothly using velocity
            self._miss_streak += 1
            if self._miss_streak <= 6 and self.state != RobotVideoState.POURING:
                self._smoothed_px = self._smoothed_px + 0.7 * self._vel_px

            # Only switch to a genuinely new foreground tree after 5 consistent frames
            # and never while BRAKING_STABILIZE or POURING.
            if (
                self.state not in (RobotVideoState.BRAKING_STABILIZE, RobotVideoState.POURING)
                and self._miss_streak >= 5
            ):
                if (
                    self._jump_candidate is not None
                    and float(np.linalg.norm(assoc_arr - self._jump_candidate)) < 90.0
                ):
                    self._jump_streak += 1
                else:
                    self._jump_candidate = assoc_arr
                    self._jump_streak = 1

                if self._jump_streak >= 4:
                    self._smoothed_px = assoc_arr
                    self._vel_px = np.zeros(2, dtype=np.float32)
                    self._jump_candidate = None
                    self._jump_streak = 0
                    self._miss_streak = 0
            elif self._miss_streak > 18:
                self._smoothed_px = None
                self._vel_px = np.zeros(2, dtype=np.float32)
                self._tree_verified = False

        # ── 2. Robot Hysteresis State Machine ─────────────────────
        if self.state == RobotVideoState.SCANNING_ROW:
            if eff_detected and self._cooldown_frames == 0:
                self._hit_streak += 1
            else:
                self._hit_streak = 0

            if self._hit_streak >= 6 and self._smoothed_px is not None:
                self.state = RobotVideoState.BRAKING_STABILIZE
                self._state_frames = 0
                self._hit_streak = 0
                self._locked_px = (int(self._smoothed_px[0]), int(self._smoothed_px[1]))

        elif self.state == RobotVideoState.BRAKING_STABILIZE:
            # Refine lock gently for 10 frames while rover brakes to 0 m/s
            if self._smoothed_px is not None and self._state_frames <= 10:
                self._locked_px = (int(self._smoothed_px[0]), int(self._smoothed_px[1]))
            if self._state_frames >= 15:
                self.state = RobotVideoState.POURING
                self._state_frames = 0

        elif self.state == RobotVideoState.POURING:
            # Pour target is FROZEN during active liquid dispensing (~0.4s = 14 frames)
            if self._state_frames >= 14:
                self.state = RobotVideoState.RETRACT_ADVANCE
                self._state_frames = 0
                self._cooldown_frames = 25  # Skip re-triggering on same tree

        elif self.state == RobotVideoState.RETRACT_ADVANCE:
            if self._state_frames >= 12:
                self.state = RobotVideoState.SCANNING_ROW
                self._state_frames = 0
                self._locked_px = None
                self._smoothed_px = None
                self._vel_px = np.zeros(2, dtype=np.float32)
                self._tree_verified = False

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
