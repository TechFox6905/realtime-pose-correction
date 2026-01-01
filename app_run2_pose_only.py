import cv2

from src.detector.person_detector import PersonDetector
from src.pose.pose_estimator import PoseEstimator
from src.utils.fps import FPSCounter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def draw_skeleton(frame, landmarks):
    """
    Draw minimal skeleton (lines only, no labels).
    Landmarks are normalized to cropped frame.
    """
    h, w, _ = frame.shape

    # Minimal connections for squat visualization
    connections = [
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ]

    for a, b in connections:
        if a not in landmarks or b not in landmarks:
            continue

        x1, y1 = landmarks[a]
        x2, y2 = landmarks[b]

        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))

        cv2.line(frame, p1, p2, (0, 255, 0), 2)


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    detector = PersonDetector()
    pose_estimator = PoseEstimator()
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
                cv2.putText(
                    frame,
                    "No person detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Run 2 — Pose Only", frame)
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
                cv2.imshow("Run 2 — Pose Only", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            landmarks = pose_result["landmarks"]

            # -----------------------------
            # Rendering
            # -----------------------------
            draw_skeleton(cropped, landmarks)

            # Put cropped frame back
            frame[y1:y2, x1:x2] = cropped

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

            cv2.imshow("Run 2 — Pose Only", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
