# src/utils/geometry.py

import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle in degrees between two 2D vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector (2D)
    v2 : np.ndarray
        Second vector (2D)

    Returns
    -------
    float
        Angle between v1 and v2 in degrees, in range [0, 180].

    Notes
    -----
    - This function is numerically stable.
    - Caller must ensure vectors are non-zero length.
    """

    # Convert to float for safety
    v1 = v1.astype(float)
    v2 = v2.astype(float)

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm_product < 1e-6:
        # Undefined angle; let caller handle
        return float("nan")

    # Clamp cosine to valid range to avoid numerical errors
    cos_theta = dot_product / norm_product
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)
