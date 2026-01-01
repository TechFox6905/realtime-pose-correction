# app.py

import cv2
import time
from collections import deque

from src.detector.person_detector import PersonDetector
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.classifier.posture_classifier import PostureClassifier
from src.feedback.feedback_engine import FeedbackEngine
from src.utils.smoothing import majority_vote
from src.utils.fps import FPSCounter


def main():
    # -----------------------------
    # Initialization (Startup only)
    # -----------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    person_detector = PersonDetector()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    classifier = PostureClassifier()
    feedback_engine = FeedbackEngine()

    fps_counter = FPSCounter()
    prediction_buffer = deque(maxlen=7)  # smoothing window

    last_feedback = ""

    # -----------------------------
    # Main Loop (Real-Time)
    # -----------------------------
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter.update()

        # -----------------------------
        # Person Detection
        # -----------------------------
        detection = person_detector.detect(frame)
        if detection is None:
            last_feedback = ""
            render(frame, fps_counter.fps, "No person detected")
            continue

        cropped_frame = detection["cropped_frame"]

        # -----------------------------
        # Pose Estimation
        # -----------------------------
        pose_result = pose_estimator.estimate(cropped_frame)
        if pose_result is None:
            render(frame, fps_counter.fps, last_feedback)
            continue

        # -----------------------------
        # Feature Extraction
        # -----------------------------
        features = feature_extractor.extract(pose_result)
        if features is None:
            render(frame, fps_counter.fps, last_feedback)
            continue

        # -----------------------------
        # Classification
        # -----------------------------
        predicted_class = classifier.predict(features)
        if predicted_class is None:
            render(frame, fps_counter.fps, last_feedback)
            continue

        prediction_buffer.append(predicted_class)
        smoothed_class = majority_vote(prediction_buffer)

        # -----------------------------
        # Feedback
        # -----------------------------
        feedback_text = feedback_engine.get_feedback(smoothed_class)
        last_feedback = feedback_text

        # -----------------------------
        # Render
        # -----------------------------
        render(frame, fps_counter.fps, feedback_text)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def render(frame, fps, feedback_text):
    """
    UI rendering only.
    No logic.
    """
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if feedback_text:
        cv2.putText(frame, feedback_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Real-Time Squat Posture Correction", frame)


if __name__ == "__main__":
    main()
