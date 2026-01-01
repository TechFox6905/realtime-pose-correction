# src/classifier/posture_classifier.py

import json
import joblib
import numpy as np
from pathlib import Path


class PostureClassifier:
    """
    Random Forest inference wrapper for squat posture classification.

    Notes:
    - Inference only (no training, no normalization).
    - Feature order and class labels are enforced via metadata.
    - Paths are configurable via constructor and are relative to project root by default.
    """

    def __init__(
        self,
        model_path: str = "models/rf_posture_classifier.joblib",
        metadata_path: str = "models/model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        # Load model once
        self.model = joblib.load(self.model_path)

        # Load and validate metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        required_keys = {"feature_order", "classes"}
        if not required_keys.issubset(self.metadata):
            raise ValueError("Invalid model metadata: missing required keys")

        self.feature_order = self.metadata["feature_order"]
        self.class_labels = set(self.metadata["classes"])

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

        # Guard feature dimensionality
        if feature_vector.shape[1] != len(self.feature_order):
            return None

        # Guard NaN / Inf values
        if not np.isfinite(feature_vector).all():
            return None

        # Inference
        prediction = self.model.predict(feature_vector)
        pred_label = str(prediction[0])

        # Validate predicted label
        if pred_label not in self.class_labels:
            return None

        return pred_label
