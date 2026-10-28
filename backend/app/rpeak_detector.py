
"""
TRACE — Experimental ECG R-Peak Detector
MEDAITHON'26 | PS-001

Detects candidate R peaks from a single-lead ECG.

Research prototype only:
- Not a validated QRS detector
- Not a cardiac diagnosis tool
- No reference annotations used during inference
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks,
)


@dataclass
class RPeakResult:
    peak_indices: list[int]
    peak_times_seconds: list[float]
    rr_intervals_seconds: list[float]
    estimated_heart_rate_bpm: float | None
    peak_count: int
    sampling_rate: float
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def detect_rpeaks(ecg, fs=250):
    signal = np.asarray(ecg, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("ECG must be one-dimensional.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if len(signal) < int(fs * 4):
        raise ValueError(
            "At least four seconds of ECG are required."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    if fs <= 40:
        raise ValueError(
            "Sampling frequency must exceed 40 Hz."
        )

    # Bandpass emphasizes QRS activity.
    sos = butter(
        3,
        [5, 18],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    # Polarity-independent QRS energy proxy.
    derivative = np.diff(
        filtered,
        prepend=filtered[0],
    )

    energy = derivative ** 2

    integration_samples = max(
        1,
        int(round(0.12 * fs)),
    )

    kernel = (
        np.ones(integration_samples)
        / integration_samples
    )

    integrated = np.convolve(
        energy,
        kernel,
        mode="same",
    )

    # Provisional adaptive threshold.
    threshold = float(
        np.median(integrated)
        + 0.5 * np.std(integrated)
    )

    minimum_distance = max(
        1,
        int(round(0.25 * fs)),
    )

    candidates, _ = find_peaks(
        integrated,
        height=threshold,
        distance=minimum_distance,
        prominence=max(
            1e-12,
            0.2 * threshold,
        ),
    )

    # Refine each candidate to a local absolute
    # peak in the filtered ECG.
    refinement_radius = max(
        1,
        int(round(0.08 * fs)),
    )

    refined = []

    for candidate in candidates:
        left = max(
            0,
            candidate - refinement_radius,
        )
        right = min(
            len(filtered),
            candidate + refinement_radius + 1,
        )

        local = np.abs(filtered[left:right])

        peak = left + int(np.argmax(local))

        refined.append(peak)

    # Remove duplicate refined positions.
    peaks = np.asarray(
        sorted(set(refined)),
        dtype=np.int64,
    )

    rr = np.diff(peaks) / fs

    if len(rr) > 0:
        median_rr = float(np.median(rr))
        heart_rate = (
            60.0 / median_rr
            if median_rr > 0
            else None
        )
    else:
        heart_rate = None

    return RPeakResult(
        peak_indices=peaks.tolist(),
        peak_times_seconds=(
            peaks / fs
        ).tolist(),
        rr_intervals_seconds=rr.tolist(),
        estimated_heart_rate_bpm=heart_rate,
        peak_count=len(peaks),
        sampling_rate=float(fs),
    )
