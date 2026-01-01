import os
import cv2
import numpy as np
from tqdm import tqdm

from src.detector.person_detector import PersonDetector
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor


DATA_DIR = "data/raw_videos"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = {
    "correct": 0,
    "knees_caving_in": 1,
    "forward_lean": 2,
}


def extract_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    detector = PersonDetector()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()

    X, y = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        if detection is None:
            feature_extractor.reset()
            continue

        pose = pose_estimator.estimate(detection["cropped_frame"])
        if pose is None:
            continue

        features = feature_extractor.extract(pose)
        if features is None:
            continue

        X.append([
            features["knee_angle"],
            features["knee_to_toe_ratio"],
            features["hip_angle"],
            features["torso_angle"],
            features["depth_ratio"],
        ])
        y.append(label)

    cap.release()
    pose_estimator.close()
    return X, y


def main():
    X_all, y_all = [], []

    for class_name, label in CLASSES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for video in tqdm(os.listdir(class_dir), desc=class_name):
            if not video.endswith(".mp4"):
                continue

            video_path = os.path.join(class_dir, video)
            X, y = extract_from_video(video_path, label)
            X_all.extend(X)
            y_all.extend(y)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_all)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_all)

    print("Saved:")
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)


if __name__ == "__main__":
    main()
