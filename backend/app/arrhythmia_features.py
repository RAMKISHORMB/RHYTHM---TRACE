
"""
TRACE — Experimental ECG Beat Feature Extraction

Uses ECG waveform samples and supplied beat positions.
Does not use annotation symbols as model inputs.

Research prototype. Not for clinical use.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


FEATURE_NAMES = [
    "rr_previous_seconds",
    "rr_next_seconds",
    "local_mean",
    "local_std",
    "local_range",
    "local_rms",
    "local_abs_mean",
    "local_median",
    "local_iqr",
    "local_max",
    "local_min",
    "peak_to_peak",
    "mean_absolute_difference",
    "max_absolute_difference",
    "waveform_0",
    "waveform_1",
    "waveform_2",
    "waveform_3",
    "waveform_4",
    "waveform_5",
    "waveform_6",
    "waveform_7",
    "waveform_8",
    "waveform_9",
    "waveform_10",
    "waveform_11",
    "waveform_12",
    "waveform_13",
    "waveform_14",
    "waveform_15",
]


def extract_beat_features(
    ecg: Sequence[float],
    beat_samples: Sequence[int],
    fs: int,
    radius_seconds: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 30 features per supplied beat.

    Returns:
        features: shape (valid_beats, 30)
        valid_indices: indices into original beat_samples

    Beat locations are supplied externally.
    No beat detection occurs in this function.
    """

    signal = np.asarray(ecg, dtype=np.float64)
    positions = np.asarray(beat_samples)

    if signal.ndim != 1 or signal.size < 3:
        raise ValueError("ECG must be a 1D signal.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite values.")

    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    if not 0 < radius_seconds <= 1.0:
        raise ValueError("Invalid beat radius.")

    if positions.ndim != 1:
        raise ValueError("Beat positions must be 1D.")

    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Beat positions must be integers.")

    if np.any(positions < 0) or np.any(
        positions >= signal.size
    ):
        raise ValueError("Beat position outside ECG.")

    if np.any(np.diff(positions) <= 0):
        raise ValueError(
            "Beat positions must be strictly increasing."
        )

    radius = max(1, round(radius_seconds * fs))

    features = []
    valid_indices = []

    for i, position in enumerate(positions):
        position = int(position)

        start = position - radius
        end = position + radius + 1

        # Require a complete neighborhood.
        if start < 0 or end > signal.size:
            continue

        local = signal[start:end]

        previous_rr = (
            (position - int(positions[i - 1])) / fs
            if i > 0
            else 0.0
        )

        next_rr = (
            (int(positions[i + 1]) - position) / fs
            if i + 1 < len(positions)
            else 0.0
        )

        differences = np.abs(np.diff(local))

        q25, q75 = np.percentile(
            local, [25, 75]
        )

        # Normalize morphology to reduce dependence
        # on absolute digital baseline and amplitude.
        centered = local - np.median(local)
        scale = max(
            float(np.percentile(np.abs(centered), 95)),
            1e-6,
        )
        normalized = centered / scale

        waveform = np.interp(
            np.linspace(0, len(normalized) - 1, 16),
            np.arange(len(normalized)),
            normalized,
        )

        row = [
            previous_rr,
            next_rr,
            float(np.mean(local)),
            float(np.std(local)),
            float(np.ptp(local)),
            float(np.sqrt(np.mean(local**2))),
            float(np.mean(np.abs(local))),
            float(np.median(local)),
            float(q75 - q25),
            float(np.max(local)),
            float(np.min(local)),
            float(np.ptp(local)),
            float(np.mean(differences)),
            float(np.max(differences)),
            *waveform.tolist(),
        ]

        features.append(row)
        valid_indices.append(i)

    if not features:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(valid_indices, dtype=np.int64),
    )
