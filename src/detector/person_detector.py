# src/detector/person_detector.py

import torch
import numpy as np


class PersonDetector:
    """
    YOLOv5-based single-person detector.

    conf_threshold: minimum YOLO confidence for accepting person detection.
    NOTE: `frame` is expected to be BGR (OpenCV default).
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        conf_threshold: float = 0.3,
        device: str | None = None,
    ):
        self.conf_threshold = conf_threshold

        # Safe device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model ONCE
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            model_name,
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # COCO person class id
        self.person_class_id = 0

    def detect(self, frame: np.ndarray):
        """
        Detect the largest person in the frame.

        Returns:
            dict with keys: bbox, cropped_frame
            or None if no person detected
        """
        # Frame validation
        if frame is None or frame.ndim != 3:
            return None

        height, width, _ = frame.shape

        # Inference
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        # Filter persons by class & confidence
        persons = [
            d for d in detections
            if int(d[5]) == self.person_class_id and d[4] >= self.conf_threshold
        ]

        if not persons:
            return None

        # Select largest bounding box (by area)
        def bbox_area(det):
            x1, y1, x2, y2 = det[:4]
            return (x2 - x1) * (y2 - y1)

        best_det = max(persons, key=bbox_area)
        x1, y1, x2, y2 = map(int, best_det[:4])

        # Clip to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = frame[y1:y2, x1:x2]

        return {
            "bbox": (x1, y1, x2, y2),
            "cropped_frame": cropped,
        }
