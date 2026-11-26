"""
Inverse Kinematics and Collision-Free Trajectory Planner for Agricultural Mobile Manipulator.
Computes smooth 3D trajectory curves (the blue arc from the reference video)
avoiding sapling tree trunks and positioning the dual-action toolhead at weed targets.
"""

import numpy as np

class ArmKinematics:
    def __init__(self):
        # Link lengths in meters matching the MuJoCo model
        self.d1 = 0.21   # Base to shoulder
        self.a2 = 0.38   # Upper arm
        self.a3 = 0.35   # Forearm
        self.d5 = 0.18   # Wrist assembly to tool tip

        # Home pose joint angles (stowed transit position)
        self.home_q = np.array([0.0, -0.6, 1.2, 0.0, -0.6, 0.0])

        # Tree trunk keep-out barrier cylinder (X, Y, radius)
        self.trunk_obstacle = {'x': 0.85, 'y': 0.45, 'radius': 0.12}

    def solve_ik_target(self, target_pos, target_pitch=-1.57):
        """
        Solves inverse kinematics for target end-effector 3D position (relative to arm base).
        target_pitch: -1.57 rad (-90 deg) points toolhead vertically downward into soil.
        """
        x, y, z = target_pos
        
        # Joint 1: Base Yaw directly pointing towards the weed target
        q1 = np.arctan2(y, x)

        # Planar distance and height relative to shoulder
        r = np.sqrt(x**2 + y**2)
        z_rel = z - self.d1

        # Wrist position offset based on desired downward tool orientation
        wrist_r = r - self.d5 * np.cos(target_pitch)
        wrist_z = z_rel - self.d5 * np.sin(target_pitch)

        # 2-link planar IK for shoulder (q2) and elbow (q3)
        D_sq = wrist_r**2 + wrist_z**2
        cos_q3 = (D_sq - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        
        # Elbow up/down configuration
        sin_q3 = -np.sqrt(max(0.0, 1.0 - cos_q3**2))
        q3 = np.arctan2(sin_q3, cos_q3)

        # Shoulder angle
        beta = np.arctan2(wrist_z, wrist_r)
        alpha = np.arctan2(self.a3 * sin_q3, self.a2 + self.a3 * cos_q3)
        q2 = beta - alpha

        # Wrist pitch to maintain vertical alignment with the ground
        q5 = target_pitch - (q2 + q3)

        # Roll joints neutral for straight weed extraction
        q4 = 0.0
        q6 = 0.0

        return np.array([q1, q2, q3, q4, q5, q6])

    def generate_trajectory_arc(self, start_pos, target_pos, num_waypoints=40, clearance_height=0.15):
        """
        Generates a smooth 3D curved trajectory (the blue glowing spline from the video),
        lifting upward first to clear obstacles, then arcing down smoothly to ground weeds.
        """
        waypoints = []
        for i in range(num_waypoints):
            s = i / float(num_waypoints - 1)
            # Cubic easing for smooth acceleration and deceleration
            t = 3 * (s**2) - 2 * (s**3)
            
            # Linear horizontal interpolation
            x = (1.0 - t) * start_pos[0] + t * target_pos[0]
            y = (1.0 - t) * start_pos[1] + t * target_pos[1]
            
            # Parabolic arc height for collision avoidance above branches/roots
            arc = 4.0 * s * (1.0 - s) * clearance_height
            z = (1.0 - t) * start_pos[2] + t * target_pos[2] + arc

            waypoints.append(np.array([x, y, z]))
            
        return waypoints

    @staticmethod
    def compute_precision_pour_target(trunk_3d, ring_offset_m: float = 0.125):
        """
        Compute exact root-zone pour point 10-15 cm (default 12.5 cm) in front
        of the trunk collar (rover-facing side) to avoid bark rot and waste.
        """
        ring_offset_m = float(np.clip(ring_offset_m, 0.10, 0.15))
        tx, ty, tz = float(trunk_3d[0]), float(trunk_3d[1]), float(trunk_3d[2])
        return np.array([tx, max(0.30, ty - ring_offset_m), tz])

    @staticmethod
    def analyze_low_branch_obstacles(detections, norm_disp, depth_localizer):
        """
        Use YOLOv8s fruit boxes + Depth-Anything-V2 disparity to measure:
          - lowest hanging fruit/branch height above ground
          - closest foreground canopy distance (for arm collision avoidance)
        """
        if not detections:
            return {"min_branch_height_m": 0.60, "canopy_front_y_m": 1.20, "low_branch_risk": False}

        h, w = norm_disp.shape[:2]
        fruit_zs, fruit_ys = [], []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            bottom_px = ((x1 + x2) // 2, min(h - 1, y2))
            _, fy, fz = depth_localizer.pixel_to_3d(bottom_px, norm_disp)
            fruit_ys.append(fy)
            fruit_zs.append(fz)

        # Estimate ground Z at bottom-center of image
        _, _, ground_z = depth_localizer.pixel_to_3d((w // 2, int(h * 0.92)), norm_disp)
        clearances = [max(0.05, fz - ground_z) for fz in fruit_zs]
        min_clearance = float(min(clearances))
        canopy_front_y = float(min(fruit_ys))

        return {
            "min_branch_height_m": round(min_clearance, 3),
            "canopy_front_y_m": round(canopy_front_y, 3),
            "low_branch_risk": min_clearance < 0.45,
        }

    def plan_collision_free_approach(self, start_pos, pour_pos, obstacle_info):
        """
        Selects a collision-free approach strategy around low-hanging branches:
          - LOW_UNDER_CANOPY: keeps arm below lowest fruit/branch clearance
          - STANDOFF_JET: stops outside canopy and angles nozzle if clearance < 0.22 m
          - STANDARD_ARC: normal parabolic clearance when canopy is high (>= 0.45 m)
        """
        min_h = obstacle_info.get("min_branch_height_m", 0.60)
        if min_h < 0.22:
            # Too low for arm entry without hitting fruit: stop 25 cm back and angle jet
            standoff = np.array([
                pour_pos[0],
                max(0.35, min(pour_pos[1] - 0.25, obstacle_info["canopy_front_y_m"] - 0.15)),
                pour_pos[2] + 0.12,
            ])
            waypoints = self.generate_trajectory_arc(start_pos, standoff, num_waypoints=25, clearance_height=0.05)
            return {
                "strategy": "STANDOFF_ANGLED_JET",
                "target_end_effector_m": np.round(standoff, 3).tolist(),
                "pour_target_m": np.round(pour_pos, 3).tolist(),
                "nozzle_pitch_rad": -0.85,  # Angled forward-downward micro-jet
                "waypoints": waypoints,
            }
        elif min_h < 0.45:
            # Low branches present: suppress upward arc so wrist stays beneath canopy
            safe_clearance = max(0.03, min(0.08, min_h - 0.12))
            waypoints = self.generate_trajectory_arc(start_pos, pour_pos, num_waypoints=30, clearance_height=safe_clearance)
            return {
                "strategy": "LOW_UNDER_CANOPY_ARC",
                "target_end_effector_m": np.round(pour_pos, 3).tolist(),
                "pour_target_m": np.round(pour_pos, 3).tolist(),
                "nozzle_pitch_rad": -1.35,
                "waypoints": waypoints,
            }
        else:
            waypoints = self.generate_trajectory_arc(start_pos, pour_pos, num_waypoints=35, clearance_height=0.15)
            return {
                "strategy": "STANDARD_CLEARANCE_ARC",
                "target_end_effector_m": np.round(pour_pos, 3).tolist(),
                "pour_target_m": np.round(pour_pos, 3).tolist(),
                "nozzle_pitch_rad": -1.57,
                "waypoints": waypoints,
            }
