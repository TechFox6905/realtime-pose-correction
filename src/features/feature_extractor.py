# src/features/feature_extractor.py

import numpy as np
from src.utils.geometry import angle_between


class FeatureExtractor:
    """
    Extracts biomechanical features for squat posture classification.

    Notes:
    - All landmarks are normalized (0–1) relative to the cropped frame.
    - This module performs geometry only (no thresholds, no ML).
    """

    def __init__(self):
        # Reference hip height captured in standing position
        self.standing_hip_y = None

    def reset(self):
        """Reset stateful calibration (e.g., when person leaves frame)."""
        self.standing_hip_y = None

    def _safe_angle(self, v1: np.ndarray, v2: np.ndarray):
        """Return angle in degrees if vectors are valid, else None."""
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return None
        return angle_between(v1, v2)

    def extract(self, pose_result: dict):
        if pose_result is None or "landmarks" not in pose_result:
            return None

        lm = pose_result["landmarks"]

        try:
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
        left_knee_angle = self._safe_angle(lh - lk, la - lk)
        right_knee_angle = self._safe_angle(rh - rk, ra - rk)
        if left_knee_angle is None or right_knee_angle is None:
            return None
        knee_angle = min(left_knee_angle, right_knee_angle)

        # -----------------------------
        # 2. Knee-to-Toe Alignment RATIO (normalized)
        # -----------------------------
        left_leg_length = np.linalg.norm(lh - la)
        right_leg_length = np.linalg.norm(rh - ra)
        leg_length = (left_leg_length + right_leg_length) / 2

        if leg_length < 1e-6:
            return None

        left_knee_to_toe = abs(lk[0] - la[0]) / leg_length
        right_knee_to_toe = abs(rk[0] - ra[0]) / leg_length
        knee_to_toe_ratio = max(left_knee_to_toe, right_knee_to_toe)

        # -----------------------------
        # 3. Hip Angle (worst side)
        # -----------------------------
        left_hip_angle = self._safe_angle(ls - lh, lk - lh)
        right_hip_angle = self._safe_angle(rs - rh, rk - rh)
        if left_hip_angle is None or right_hip_angle is None:
            return None
        hip_angle = min(left_hip_angle, right_hip_angle)

        # -----------------------------
        # 4. Torso Inclination Angle
        # -----------------------------
        mid_hip = (lh + rh) / 2
        mid_shoulder = (ls + rs) / 2
        torso_vector = mid_shoulder - mid_hip
        vertical_axis = np.array([0, -1])

        torso_angle = self._safe_angle(torso_vector, vertical_axis)
        if torso_angle is None:
            return None

        # -----------------------------
        # 5. Squat Depth Ratio (calibrated)
        # -----------------------------
        current_hip_y = mid_hip[1]

        # Capture standing height only when torso is upright
        if self.standing_hip_y is None and torso_angle < 10:
            self.standing_hip_y = current_hip_y

        if self.standing_hip_y is None:
            return None

        depth_ratio = (current_hip_y - self.standing_hip_y) / leg_length

        return {
            "knee_angle": float(knee_angle),
            "knee_to_toe_ratio": float(knee_to_toe_ratio),
            "hip_angle": float(hip_angle),
            "torso_angle": float(torso_angle),
            "depth_ratio": float(depth_ratio),
        }
