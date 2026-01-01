# src/detector/person_detector.py

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """
    YOLOv8n-based single-person detector (CPU optimized).

    NOTE:
    - Input frame expected in BGR format (OpenCV default)
    - Output contract identical to YOLOv5 version
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load model ONCE
        self.model = YOLO(model_path)

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

        # YOLOv8 inference (GPU-Enabled)
        results = self.model(
        frame,
        device=0,          # 👈 FORCE CUDA (RTX 4050)
        imgsz=640,
        conf=self.conf_threshold,
        iou=self.iou_threshold,
        verbose=False,
    )


        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes

        persons = []
        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id == self.person_class_id:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                persons.append((x1, y1, x2, y2))

        if not persons:
            return None

        # Select largest bounding box
        def area(b):
            return (b[2] - b[0]) * (b[3] - b[1])

        x1, y1, x2, y2 = map(int, max(persons, key=area))

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
