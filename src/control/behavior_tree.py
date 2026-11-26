"""
Behavior Tree State Machine for Dual-Action Weeding and Precision Irrigation.
Orchestrates the 5-step mission sequence observed in gemini_generated_video_9aeb227e.mp4:
1. APPROACH_SAPLING
2. LOCK_BOUNDING_BOX
3. EXECUTE_TRAJECTORY_ARC
4. ROTARY_WEEDING (Mechanical Grass Removal -> "CLEARED")
5. MICRO_IRRIGATION (0.25 L/s Targeted Root Spray)
6. STOW_AND_ADVANCE
"""

import time
from enum import Enum

class MissionState(Enum):
    APPROACH = 1
    PERCEPTION_LOCK = 2
    TRAJECTORY_ARC = 3
    ROTARY_WEEDING = 4
    MICRO_IRRIGATION = 5
    STOW_AND_COMPLETE = 6

class MissionBehaviorTree:
    def __init__(self):
        self.state = MissionState.APPROACH
        self.state_start_time = 0.0
        self.current_time = 0.0

        # Telemetry metrics
        self.weeds_cleared = False
        self.irrigation_active = False
        self.flow_rate = 0.0  # L/s
        self.total_dispensed_liters = 0.0
        self.system_status = "INITIALIZING"
        self.efficiency = 99.8

    def update(self, sim_time, is_at_tree=True):
        self.current_time = sim_time
        elapsed_in_state = self.current_time - self.state_start_time

        if self.state == MissionState.APPROACH:
            self.system_status = "NAVIGATING TO ROW NODE 04"
            if is_at_tree and elapsed_in_state > 1.0:
                self._transition(MissionState.PERCEPTION_LOCK)

        elif self.state == MissionState.PERCEPTION_LOCK:
            self.system_status = "LOCKING 3D BOUNDING VOLUME"
            if elapsed_in_state > 1.2:
                self._transition(MissionState.TRAJECTORY_ARC)

        elif self.state == MissionState.TRAJECTORY_ARC:
            self.system_status = "COMPUTING COLLISION-FREE IK ARC"
            if elapsed_in_state > 2.0:
                self._transition(MissionState.ROTARY_WEEDING)

        elif self.state == MissionState.ROTARY_WEEDING:
            self.system_status = "ACTIVE MECHANICAL CULTIVATION"
            if elapsed_in_state > 2.5:
                self.weeds_cleared = True
                self._transition(MissionState.MICRO_IRRIGATION)

        elif self.state == MissionState.MICRO_IRRIGATION:
            self.system_status = "TARGETED ROOT HYDRATION"
            self.irrigation_active = True
            self.flow_rate = 0.25
            self.total_dispensed_liters += self.flow_rate * 0.002
            if elapsed_in_state > 1.8:
                self.irrigation_active = False
                self.flow_rate = 0.0
                self._transition(MissionState.STOW_AND_COMPLETE)

        elif self.state == MissionState.STOW_AND_COMPLETE:
            self.system_status = "SEQUENCE COMPLETE - READY"

        return {
            'state': self.state.name,
            'status_label': self.system_status,
            'weeds_cleared': self.weeds_cleared,
            'irrigation_active': self.irrigation_active,
            'flow_rate': self.flow_rate,
            'efficiency': self.efficiency
        }

    def _transition(self, new_state):
        self.state = new_state
        self.state_start_time = self.current_time
