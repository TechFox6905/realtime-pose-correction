# src/pose/pose_estimator.py

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    MediaPipe Pose wrapper.
    Extracts normalized pose landmarks from a cropped person frame.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_threshold: float = 0.5,
    ):
        self.visibility_threshold = visibility_threshold

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmark_enum = mp.solutions.pose.PoseLandmark

        # Landmarks required for squat analysis
        self.required_landmarks = [
            self.landmark_enum.LEFT_HIP,
            self.landmark_enum.RIGHT_HIP,
            self.landmark_enum.LEFT_KNEE,
            self.landmark_enum.RIGHT_KNEE,
            self.landmark_enum.LEFT_ANKLE,
            self.landmark_enum.RIGHT_ANKLE,
            self.landmark_enum.LEFT_SHOULDER,
            self.landmark_enum.RIGHT_SHOULDER,
        ]

    def estimate(self, frame: np.ndarray):
        """
        Estimate pose landmarks from a cropped BGR frame.

        Returns:
            dict with key: "landmarks"
            or None if pose is not reliably detected
        """
        if frame is None or frame.ndim != 3:
            return None

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        extracted = {}

        for lm_id in self.required_landmarks:
            lm = landmarks[lm_id.value]

            if lm.visibility < self.visibility_threshold:
                return None

            extracted[lm_id.name] = (lm.x, lm.y)

        return {
            "landmarks": extracted
        }
