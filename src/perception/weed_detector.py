"""
Perception Module: 3D Bounding Box & Target Extraction for Agricultural Weeding.
Identifies ground weed clusters, calculates 3D bounding volumes (the green HUD box),
and computes the safety keep-out envelope around the pomegranate sapling trunk.
"""

import numpy as np

class WeedPerceptionSystem:
    def __init__(self, trunk_pos=np.array([0.85, 0.45, 0.0])):
        self.trunk_pos = trunk_pos
        self.trunk_safety_radius = 0.08  # 8 cm critical keep-out barrier for bark

    def detect_weed_cluster(self, weed_ground_offset=np.array([-0.12, 0.08, 0.02])):
        """
        Simulates camera detection (YOLO-World + 3D Depth Disparity).
        Returns target weeding coordinate and 3D bounding box dimensions.
        """
        # Actual world position of the weed cluster
        target_world_pos = self.trunk_pos + weed_ground_offset
        
        # Bounding box corners around the target area
        box_size = np.array([0.30, 0.30, 0.25])
        box_center = target_world_pos + np.array([0.0, 0.0, box_size[2] / 2.0])

        # Safety verification: ensure weed centroid is outside the tree bark
        dist_to_trunk = np.linalg.norm(target_world_pos[:2] - self.trunk_pos[:2])
        is_safe = dist_to_trunk > self.trunk_safety_radius

        return {
            'target_pos': target_world_pos,
            'box_center': box_center,
            'box_size': box_size,
            'dist_to_trunk': dist_to_trunk,
            'safe_for_actuation': is_safe
        }
