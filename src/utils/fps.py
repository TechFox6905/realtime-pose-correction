# src/utils/fps.py

import time


class FPSCounter:
    """
    Simple FPS counter for real-time video loops.
    """

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        current_time = time.time()
        delta = current_time - self.prev_time

        if delta > 0:
            self.fps = 1.0 / delta

        self.prev_time = current_time
