
"""
TRACE — Reference-Free ECG Quality Features
MEDAITHON PS-001

Experimental engineering features only.
Not a clinical ECG interpretation algorithm.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class ECGQualityResult:
    baseline_rms: float
    baseline_ratio: float
    max_step: float
    step_to_median_ratio: float
    longest_flat_duration_ms: float
    flatline_fraction: float
    quality_flags: list[str]
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def analyze_ecg_quality(ecg, fs=250):
    signal = np.asarray(ecg, dtype=np.float64)

    if signal.ndim != 1 or len(signal) < 100:
        raise ValueError("ECG must contain at least 100 samples.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    duration = len(signal) / fs

    if duration < 2:
        raise ValueError("At least 2 seconds of ECG required.")

    # Low-frequency baseline estimate.
    sos = butter(
        2,
        0.5,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    baseline = sosfiltfilt(sos, signal)

    baseline_centered = baseline - np.mean(baseline)
    ecg_centered = signal - np.mean(signal)

    baseline_rms = float(
        np.sqrt(np.mean(baseline_centered ** 2))
    )

    ecg_rms = float(
        np.sqrt(np.mean(ecg_centered ** 2))
    )

    baseline_ratio = float(
        baseline_rms / max(ecg_rms, 1e-9)
    )

    differences = np.abs(np.diff(signal))

    max_step = float(np.max(differences))

    median_step = float(np.median(differences))

    step_to_median_ratio = float(
        max_step / max(median_step, 1e-6)
    )

    # Identify long runs of nearly unchanged samples.
    # Threshold is provisional and relative to
    # the signal's peak-to-peak amplitude.
    signal_range = float(np.ptp(signal))

    flat_threshold = max(
        1e-9,
        signal_range * 0.0005,
    )

    flat_mask = differences < flat_threshold

    longest_run = 0
    current_run = 0

    for is_flat in flat_mask:
        if is_flat:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 0

    longest_flat_duration_ms = float(
        longest_run / fs * 1000
    )

    flatline_fraction = float(
        np.mean(flat_mask)
    )

    flags = []

    # Provisional thresholds, not validated
    # on independent ECG recordings.
    if baseline_ratio > 0.45:
        flags.append("POSSIBLE_BASELINE_DRIFT")

    if step_to_median_ratio > 100:
        flags.append("POSSIBLE_ABRUPT_CHANGE")

    if longest_flat_duration_ms >= 150:
        flags.append("POSSIBLE_LOCAL_DROPOUT")

    return ECGQualityResult(
        baseline_rms=baseline_rms,
        baseline_ratio=baseline_ratio,
        max_step=max_step,
        step_to_median_ratio=step_to_median_ratio,
        longest_flat_duration_ms=longest_flat_duration_ms,
        flatline_fraction=flatline_fraction,
        quality_flags=flags,
    )
