
"""
TRACE — Motion Activity Intelligence
PS-001: Motion-Compensated Wearable ECG

Experimental signal activity measurements.
Supports a 1D simulated motion proxy or a 3-axis IMU.

Motion labels are heuristic, not validated activity classes.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class MotionAnalysis:
    rms: float
    peak: float
    std: float
    activity_score: float
    activity_level: str
    channels: int
    source: str
    experimental: bool = True

    def to_dict(self):
        return asdict(self)


def analyze_motion(
    imu,
    source="simulated_imu",
    low_threshold=0.05,
    high_threshold=0.25,
):
    data = np.asarray(imu, dtype=np.float64)

    if data.ndim == 1:
        data = data[:, None]

    if data.ndim != 2 or data.shape[1] not in (1, 3):
        raise ValueError(
            "IMU must have shape (samples, 1 or 3)."
        )

    if len(data) < 2:
        raise ValueError("Insufficient motion samples.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Motion contains invalid values.")

    if not (0 <= low_threshold < high_threshold):
        raise ValueError("Invalid activity thresholds.")

    # Remove each channel's constant offset.
    # This reduces sensitivity to a fixed sensor bias
    # or the gravity component in a stationary IMU.
    centered = data - np.mean(data, axis=0, keepdims=True)

    # Motion magnitude from the centered channels.
    magnitude = np.linalg.norm(centered, axis=1)

    rms = float(np.sqrt(np.mean(magnitude ** 2)))
    peak = float(np.max(magnitude))
    std = float(np.std(magnitude))

    # Heuristic score in the units of the input signal.
    # This is NOT a calibrated activity probability.
    activity_score = rms

    if activity_score < low_threshold:
        activity_level = "LOW"
    elif activity_score < high_threshold:
        activity_level = "MODERATE"
    else:
        activity_level = "HIGH"

    return MotionAnalysis(
        rms=rms,
        peak=peak,
        std=std,
        activity_score=activity_score,
        activity_level=activity_level,
        channels=int(data.shape[1]),
        source=source,
    )
