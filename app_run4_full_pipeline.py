import cv2

from src.detector.person_detector import PersonDetector
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.classifier.posture_classifier import PostureClassifier
from src.feedback.feedback_engine import FeedbackEngine
from src.utils.fps import FPSCounter
from src.utils.smoothing import MajorityVoteSmoother
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    detector = PersonDetector()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    classifier = PostureClassifier()
    feedback_engine = FeedbackEngine()
    smoother = MajorityVoteSmoother(window_size=7)
    fps_counter = FPSCounter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter.update()

            # -----------------------------
            # Person Detection
            # -----------------------------
            detection = detector.detect(frame)
            if detection is None:
                feature_extractor.reset()
                smoother.reset()
                feedback = ""
                cv2.putText(
                    frame,
                    "No person detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Run 4 — Full Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            x1, y1, x2, y2 = detection["bbox"]
            cropped = detection["cropped_frame"]

            # -----------------------------
            # Pose Estimation
            # -----------------------------
            pose_result = pose_estimator.estimate(cropped)
            if pose_result is None:
                cv2.imshow("Run 4 — Full Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # -----------------------------
            # Feature Extraction
            # -----------------------------
            features = feature_extractor.extract(pose_result)
            if features is None:
                cv2.imshow("Run 4 — Full Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # -----------------------------
            # Classification
            # -----------------------------
            raw_class = classifier.predict(features)
            smooth_class = smoother.update(raw_class)

            # -----------------------------
            # Feedback
            # -----------------------------
            feedback = feedback_engine.get_feedback(smooth_class)

            # -----------------------------
            # Rendering
            # -----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"FPS: {int(fps_counter.fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if feedback:
                cv2.putText(
                    frame,
                    feedback,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )

            cv2.imshow("Run 4 — Full Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
