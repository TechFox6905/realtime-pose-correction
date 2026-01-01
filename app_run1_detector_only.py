# app_run1_detector_only.py

import cv2
from src.detector.person_detector import PersonDetector
from src.utils.fps import FPSCounter


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    detector = PersonDetector()
    fps_counter = FPSCounter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter.update()

            detection = detector.detect(frame)

            if detection is not None:
                x1, y1, x2, y2 = detection["bbox"]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            # FPS overlay
            cv2.putText(
                frame,
                f"FPS: {int(fps_counter.fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Run 1 — Detector Only", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
