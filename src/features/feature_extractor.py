# src/features/feature_extractor.py

import numpy as np
from src.utils.geometry import angle_between


class FeatureExtractor:
    """
    Extracts biomechanical features for squat posture classification.
    """

    def __init__(self):
        # Reference hip height captured in standing position
        self.standing_hip_y = None

    def extract(self, pose_result: dict):
        if pose_result is None or "landmarks" not in pose_result:
            return None

        lm = pose_result["landmarks"]

        try:
            # Required landmarks
            lh = np.array(lm["LEFT_HIP"])
            rh = np.array(lm["RIGHT_HIP"])
            lk = np.array(lm["LEFT_KNEE"])
            rk = np.array(lm["RIGHT_KNEE"])
            la = np.array(lm["LEFT_ANKLE"])
            ra = np.array(lm["RIGHT_ANKLE"])
            ls = np.array(lm["LEFT_SHOULDER"])
            rs = np.array(lm["RIGHT_SHOULDER"])
        except KeyError:
            return None

        # -----------------------------
        # 1. Knee Angle (min of both)
        # -----------------------------
        left_knee_angle = angle_between(lh - lk, la - lk)
        right_knee_angle = angle_between(rh - rk, ra - rk)
        knee_angle = min(left_knee_angle, right_knee_angle)

        # -----------------------------
        # 2. Knee-to-Toe Alignment Ratio
        # -----------------------------
        left_knee_to_toe = abs(lk[0] - la[0])
        right_knee_to_toe = abs(rk[0] - ra[0])
        knee_to_toe_ratio = max(left_knee_to_toe, right_knee_to_toe)

        # -----------------------------
        # 3. Hip Angle
        # -----------------------------
        left_hip_angle = angle_between(ls - lh, lk - lh)
        right_hip_angle = angle_between(rs - rh, rk - rh)
        hip_angle = min(left_hip_angle, right_hip_angle)

        # -----------------------------
        # 4. Torso Inclination Angle
        # -----------------------------
        mid_hip = (lh + rh) / 2
        mid_shoulder = (ls + rs) / 2
        torso_vector = mid_shoulder - mid_hip
        vertical_axis = np.array([0, -1])
        torso_angle = angle_between(torso_vector, vertical_axis)

        # -----------------------------
        # 5. Squat Depth Ratio
        # -----------------------------
        current_hip_y = mid_hip[1]

        if self.standing_hip_y is None:
            self.standing_hip_y = current_hip_y

        left_leg_length = np.linalg.norm(lh - la)
        right_leg_length = np.linalg.norm(rh - ra)
        leg_length = (left_leg_length + right_leg_length) / 2

        if leg_length == 0:
            return None

        depth_ratio = (current_hip_y - self.standing_hip_y) / leg_length

        return {
            "knee_angle": float(knee_angle),
            "knee_to_toe_ratio": float(knee_to_toe_ratio),
            "hip_angle": float(hip_angle),
            "torso_angle": float(torso_angle),
            "depth_ratio": float(depth_ratio),
        }
