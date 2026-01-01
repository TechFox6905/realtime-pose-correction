from collections import deque, Counter
from typing import Optional


class MajorityVoteSmoother:
    """
    Temporal smoother using majority voting over last N predictions.

    - Ignores None values
    - Deterministic
    - Stateless ML-wise (pure logic)
    """

    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def reset(self):
        """Clear prediction history (e.g., when person leaves frame)."""
        self.buffer.clear()

    def update(self, prediction: Optional[str]) -> Optional[str]:
        """
        Add new prediction and return smoothed result.

        Args:
            prediction: class label or None

        Returns:
            Smoothed class label or None
        """
        if prediction is not None:
            self.buffer.append(prediction)

        if not self.buffer:
            return None

        counts = Counter(self.buffer)
        return counts.most_common(1)[0][0]
