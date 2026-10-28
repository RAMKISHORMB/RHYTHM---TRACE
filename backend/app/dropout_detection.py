
"""
TRACE — Localized ECG Dropout Detection
MEDAITHON'26 | PS-001

Reference-free, experimental ECG signal-quality module.

Detects sustained low-amplitude ECG segments relative
to the amplitude distribution of the observed recording.

IMPORTANT:
- Does not require a clean reference ECG.
- Does not diagnose cardiac conditions.
- Does not confirm electrode disconnection.
- Thresholds are provisional and synthetic-data-derived.
- Not clinically validated.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DropoutResult:
    reference_amplitude: float
    minimum_local_amplitude: float
    minimum_amplitude_ratio: float
    longest_low_amplitude_ms: float
    low_amplitude_fraction: float
    possible_dropout: bool
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def detect_local_dropout(
    ecg,
    fs=250,
    frame_ms=100,
    hop_ms=20,
    relative_threshold=0.15,
    min_duration_ms=300,
):
    """
    Detect possible localized ECG dropout.

    Parameters
    ----------
    ecg:
        One-dimensional observed ECG signal.
    fs:
        ECG sampling frequency in Hz.
    frame_ms:
        Analysis frame duration in milliseconds.
    hop_ms:
        Time between consecutive frames.
    relative_threshold:
        Fraction of reference amplitude below which
        a frame is considered low amplitude.
    min_duration_ms:
        Minimum sustained low-amplitude duration
        required for a possible-dropout warning.

    Returns
    -------
    DropoutResult
        Experimental signal-quality measurements.
    """

    signal = np.asarray(ecg, dtype=np.float64)

    # -----------------------------
    # Input validation
    # -----------------------------

    if signal.ndim != 1 or len(signal) < 100:
        raise ValueError(
            "ECG must be a 1D array containing "
            "at least 100 samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "ECG contains non-finite values."
        )

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling frequency must be positive."
        )

    if not (
        np.isfinite(frame_ms)
        and np.isfinite(hop_ms)
        and np.isfinite(min_duration_ms)
        and frame_ms > 0
        and hop_ms > 0
        and min_duration_ms > 0
        and 0 < relative_threshold < 1
    ):
        raise ValueError(
            "Invalid dropout detection parameters."
        )

    frame = max(
        2,
        int(round(fs * frame_ms / 1000))
    )

    hop = max(
        1,
        int(round(fs * hop_ms / 1000))
    )

    if frame > len(signal):
        raise ValueError(
            "Analysis frame exceeds ECG length."
        )

    # -----------------------------
    # Local amplitude measurement
    # -----------------------------

    starts = np.arange(
        0,
        len(signal) - frame + 1,
        hop,
    )

    amplitudes = []

    for start in starts:
        segment = signal[start:start + frame]

        centered = segment - np.mean(segment)

        local_rms = np.sqrt(
            np.mean(centered ** 2)
        )

        amplitudes.append(local_rms)

    amplitudes = np.asarray(
        amplitudes,
        dtype=np.float64,
    )

    # Use the 75th percentile as an internal
    # amplitude reference.
    reference_amplitude = float(
        np.percentile(amplitudes, 75)
    )

    minimum_local_amplitude = float(
        np.min(amplitudes)
    )

    # -----------------------------
    # Handle near-constant ECG
    # -----------------------------

    if reference_amplitude <= 1e-9:
        duration_ms = float(
            len(signal) / fs * 1000
        )

        return DropoutResult(
            reference_amplitude=reference_amplitude,
            minimum_local_amplitude=minimum_local_amplitude,
            minimum_amplitude_ratio=0.0,
            longest_low_amplitude_ms=duration_ms,
            low_amplitude_fraction=1.0,
            possible_dropout=bool(
                duration_ms >= min_duration_ms
            ),
        )

    # -----------------------------
    # Relative amplitude analysis
    # -----------------------------

    ratios = amplitudes / reference_amplitude

    low_amplitude_mask = (
        ratios < relative_threshold
    )

    # -----------------------------
    # Find longest continuous run
    # -----------------------------

    longest_run = 0
    current_run = 0

    for is_low in low_amplitude_mask:
        if is_low:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 0

    # Frames overlap, so a run spans approximately
    # one full frame plus the hop distances.
    if longest_run > 0:
        longest_duration_ms = (
            frame_ms
            + (longest_run - 1) * hop_ms
        )
    else:
        longest_duration_ms = 0.0

    # -----------------------------
    # Experimental decision
    # -----------------------------

    possible_dropout = bool(
        longest_duration_ms >= min_duration_ms
    )

    return DropoutResult(
        reference_amplitude=reference_amplitude,
        minimum_local_amplitude=minimum_local_amplitude,
        minimum_amplitude_ratio=float(
            np.min(ratios)
        ),
        longest_low_amplitude_ms=float(
            longest_duration_ms
        ),
        low_amplitude_fraction=float(
            np.mean(low_amplitude_mask)
        ),
        possible_dropout=possible_dropout,
    )
