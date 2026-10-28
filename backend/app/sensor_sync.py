
"""
TRACE — Experimental ECG/IMU Synchronization

Aligns supplied ECG and motion samples on a common
relative time axis.

Supports 1D motion proxies and multi-axis IMU signals.

No physical sensor synchronization is claimed.
Research prototype; not for clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SynchronizedWindow:
    timestamps: np.ndarray
    ecg: np.ndarray
    imu: np.ndarray
    ecg_fs: float
    imu_fs: float
    imu_channels: int
    source: str
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamps": self.timestamps.tolist(),
            "ecg": self.ecg.tolist(),
            "imu": self.imu.tolist(),
            "ecg_fs": self.ecg_fs,
            "imu_fs": self.imu_fs,
            "imu_channels": self.imu_channels,
            "source": self.source,
            "experimental": self.experimental,
            "clinical_validated": self.clinical_validated,
        }


def synchronize_sensors(
    ecg: Sequence[float],
    imu: Sequence[float] | np.ndarray,
    ecg_fs: float,
    imu_fs: float,
    source: str = "simulated_imu",
) -> SynchronizedWindow:
    """
    Align uniformly sampled ECG and IMU arrays.

    Assumes both arrays start at the same time (t=0).
    This assumption must be established externally.

    No timestamp offset or clock drift estimation
    is performed.
    """

    ecg_array = np.asarray(ecg, dtype=np.float64)
    imu_array = np.asarray(imu, dtype=np.float64)

    if ecg_array.ndim != 1 or len(ecg_array) < 2:
        raise ValueError("ECG must be a 1D signal.")

    if imu_array.ndim == 1:
        imu_array = imu_array[:, None]

    if imu_array.ndim != 2:
        raise ValueError(
            "IMU must have shape (samples, channels)."
        )

    if len(imu_array) < 2:
        raise ValueError(
            "IMU must contain at least two samples."
        )

    if imu_array.shape[1] not in (1, 3):
        raise ValueError(
            "IMU must contain either 1 or 3 channels."
        )

    if not np.isfinite(ecg_fs) or ecg_fs <= 0:
        raise ValueError("Invalid ECG sampling rate.")

    if not np.isfinite(imu_fs) or imu_fs <= 0:
        raise ValueError("Invalid IMU sampling rate.")

    if not np.all(np.isfinite(ecg_array)):
        raise ValueError("ECG contains non-finite values.")

    if not np.all(np.isfinite(imu_array)):
        raise ValueError("IMU contains non-finite values.")

    ecg_times = (
        np.arange(len(ecg_array), dtype=np.float64)
        / ecg_fs
    )

    imu_times = (
        np.arange(len(imu_array), dtype=np.float64)
        / imu_fs
    )

    # Refuse extrapolation beyond the IMU recording.
    if ecg_times[-1] > imu_times[-1] + 1e-9:
        raise ValueError(
            "IMU does not cover the full ECG time range."
        )

    aligned_imu = np.column_stack([
        np.interp(
            ecg_times,
            imu_times,
            imu_array[:, channel],
        )
        for channel in range(imu_array.shape[1])
    ])

    return SynchronizedWindow(
        timestamps=ecg_times,
        ecg=ecg_array,
        imu=aligned_imu,
        ecg_fs=float(ecg_fs),
        imu_fs=float(imu_fs),
        imu_channels=int(imu_array.shape[1]),
        source=source,
    )
