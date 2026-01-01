# src/classifier/posture_classifier.py

import json
import joblib
import numpy as np
from pathlib import Path


class PostureClassifier:
    """
    Random Forest inference wrapper for squat posture classification.
    """

    def __init__(
        self,
        model_path: str = "models/rf_posture_classifier.joblib",
        metadata_path: str = "models/model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        # Load model and metadata once
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_order = self.metadata["feature_order"]
        self.class_labels = self.metadata["classes"]

    def predict(self, features: dict):
        """
        Predict posture class from feature dictionary.

        Returns:
            str (class label) or None on failure
        """
        if features is None:
            return None

        try:
            # Enforce feature order
            feature_vector = np.array(
                [features[name] for name in self.feature_order],
                dtype=float
            ).reshape(1, -1)
        except KeyError:
            return None

        # Inference
        prediction = self.model.predict(feature_vector)

        return str(prediction[0])
