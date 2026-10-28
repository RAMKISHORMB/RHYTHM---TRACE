
"""
TRACE — Stage 11.8
Adaptive Motion Artefact Cancellation

Normalized Least Mean Squares (NLMS).

Uses an independent simulated motion reference.
Research prototype; not clinically validated.
"""

import numpy as np


def nlms_motion_cancel(
    observed_ecg,
    motion_reference,
    taps=12,
    step_size=0.05,
    regularization=0.01,
    fs=250,
):
    ecg = np.asarray(observed_ecg, dtype=np.float64)
    motion = np.asarray(motion_reference, dtype=np.float64)

    if ecg.ndim != 1 or motion.ndim != 1:
        raise ValueError("Both inputs must be 1D.")

    if len(ecg) != len(motion):
        raise ValueError("Input lengths must match.")

    if len(ecg) < max(taps, fs):
        raise ValueError("Not enough ECG samples.")

    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains non-finite values.")

    if not np.isfinite(motion).all():
        raise ValueError("Motion contains non-finite values.")

    if not isinstance(taps, int) or taps < 1:
        raise ValueError("Invalid filter taps.")

    if not 0 < step_size < 2:
        raise ValueError("Invalid step size.")

    if regularization <= 0:
        raise ValueError("Invalid regularization.")

    # Remove large ECG DC offset using a robust
    # recording-level median.
    # This is an offline research implementation.
    baseline = float(np.median(ecg))

    centered_ecg = ecg - baseline
    centered_motion = motion - np.mean(motion)

    weights = np.zeros(taps, dtype=np.float64)
    history = np.zeros(taps, dtype=np.float64)

    estimated_noise = np.zeros_like(ecg)
    compensated_centered = np.zeros_like(ecg)

    for i in range(len(ecg)):
        history[1:] = history[:-1]
        history[0] = centered_motion[i]

        noise_estimate = float(
            np.dot(weights, history)
        )

        error = centered_ecg[i] - noise_estimate

        power = float(
            np.dot(history, history)
        )

        if i >= taps - 1:
            weights += (
                step_size
                * error
                * history
                / (regularization + power)
            )

        estimated_noise[i] = noise_estimate
        compensated_centered[i] = error

    # Restore the original signal baseline.
    compensated_ecg = compensated_centered + baseline

    return {
        "compensated_ecg": compensated_ecg,
        "estimated_noise": estimated_noise,
        "final_weights": weights.copy(),
        "method": "NLMS",
        "motion_source": "simulated",
        "processing_mode": "offline",
        "clinical_validated": False,
    }
