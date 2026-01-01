# src/feedback/feedback_engine.py


class FeedbackEngine:
    """
    Deterministic mapping from posture class to feedback text.
    """

    def __init__(self):
        # Locked feedback mapping for MVP v1
        self._feedback_map = {
            "correct": "Good squat form",
            "knees_caving_in": "Push your knees outward",
            "forward_lean": "Keep your chest upright",
        }

    def get_feedback(self, predicted_class: str) -> str:
        """
        Return feedback text for predicted posture class.

        Returns empty string if class is None or unknown.
        """
        if not predicted_class:
            return ""

        return self._feedback_map.get(predicted_class, "")
