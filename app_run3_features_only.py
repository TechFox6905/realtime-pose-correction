import cv2

from src.detector.person_detector import PersonDetector
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.utils.fps import FPSCounter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    detector = PersonDetector()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
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
                print("No person detected")
                cv2.imshow("Run 3 — Features Only", frame)
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
                cv2.imshow("Run 3 — Features Only", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # -----------------------------
            # Feature Extraction
            # -----------------------------
            features = feature_extractor.extract(pose_result)
            if features is None:
                print("Feature extraction failed")
                cv2.imshow("Run 3 — Features Only", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # -----------------------------
            # PRINT FEATURES (CORE OF RUN 3)
            # -----------------------------
            print(
                f"FPS: {int(fps_counter.fps)} | "
                f"KneeAngle: {features['knee_angle']:.1f} | "
                f"KneeToe: {features['knee_to_toe_ratio']:.3f} | "
                f"HipAngle: {features['hip_angle']:.1f} | "
                f"TorsoAngle: {features['torso_angle']:.1f} | "
                f"DepthRatio: {features['depth_ratio']:.3f}"
            )

            # Basic rendering (box only)
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

            cv2.imshow("Run 3 — Features Only", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
