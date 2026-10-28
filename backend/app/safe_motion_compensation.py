
"""
TRACE — Stage 11.9
Conservative Motion-Aware Compensation with Fallback

Research prototype only.

Uses lagged motion-reference regression to estimate
motion-correlated ECG interference.

Important:
- No clean ECG is used by the compensation function.
- No physical accelerometer is involved.
- Acceptance checks are heuristic, not proof of safety.
- No clinical validation.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def calculate_rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def highpass_for_estimation(signal, fs=250):
    """
    Remove slow ECG baseline for coefficient estimation.

    The returned signal is used only for fitting.
    The original ECG is preserved for output.
    """
    sos = butter(
        2,
        0.5,
        btype="highpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def make_lagged_motion_matrix(motion, taps=5):
    """
    Build a causal-lag feature matrix.

    Column 0: current motion sample
    Column 1: previous motion sample
    etc.
    """
    n = len(motion)
    matrix = np.zeros((n, taps), dtype=np.float64)

    for lag in range(taps):
        if lag == 0:
            matrix[:, lag] = motion
        else:
            matrix[lag:, lag] = motion[:-lag]

    return matrix


def safe_motion_compensate(
    observed_ecg,
    motion_reference,
    fs=250,
    taps=5,
    ridge=10.0,
    max_correction_fraction=0.08,
    min_motion_rms=0.02,
    min_explained_variance=0.01,
):
    """
    Attempt conservative motion-reference correction.

    Returns original ECG when checks fail.

    No ground-truth ECG is required at runtime.
    """

    ecg = np.asarray(observed_ecg, dtype=np.float64)
    motion = np.asarray(motion_reference, dtype=np.float64)

    if ecg.ndim != 1 or motion.ndim != 1:
        raise ValueError("Inputs must be 1D.")

    if len(ecg) != len(motion):
        raise ValueError("ECG/motion length mismatch.")

    if len(ecg) < max(fs * 4, taps * 4):
        raise ValueError("Insufficient signal duration.")

    if not np.isfinite(ecg).all():
        raise ValueError("Invalid ECG samples.")

    if not np.isfinite(motion).all():
        raise ValueError("Invalid motion samples.")

    if fs <= 1 or taps < 1 or ridge <= 0:
        raise ValueError("Invalid filter parameters.")

    if not 0 < max_correction_fraction < 1:
        raise ValueError("Invalid correction limit.")

    baseline = float(np.median(ecg))
    ecg_centered = ecg - baseline

    motion_centered = motion - np.mean(motion)
    motion_rms = calculate_rms(motion_centered)

    def fallback(reason, explained=0.0, correction=0.0):
        return {
            "output_ecg": ecg.copy(),
            "candidate_ecg": None,
            "accepted": False,
            "decision": "PRESERVE_ORIGINAL",
            "reason": reason,
            "motion_rms": motion_rms,
            "explained_variance": float(explained),
            "correction_fraction": float(correction),
            "method": "motion_reference_regression",
            "clinical_validated": False,
        }

    if motion_rms < min_motion_rms:
        return fallback("MOTION_REFERENCE_TOO_WEAK")

    features = make_lagged_motion_matrix(
        motion_centered,
        taps=taps,
    )

    target = highpass_for_estimation(
        ecg_centered,
        fs=fs,
    )

    # Exclude the initial lagged-feature warmup.
    X = features[taps:]
    y = target[taps:]

    # Ridge-regularized least squares.
    gram = X.T @ X
    gram += ridge * np.eye(taps)

    coefficients = np.linalg.solve(
        gram,
        X.T @ y,
    )

    estimated_noise = features @ coefficients

    if not np.isfinite(estimated_noise).all():
        return fallback("INVALID_NOISE_ESTIMATE")

    # Remove any estimated constant offset.
    estimated_noise -= np.mean(estimated_noise)

    target_power = calculate_rms(target)

    if target_power < 1e-10:
        return fallback("ECG_ESTIMATION_SIGNAL_TOO_WEAK")

    residual = target - estimated_noise

    explained_variance = (
        1.0
        - np.mean(residual ** 2)
        / max(np.mean(target ** 2), 1e-12)
    )

    if explained_variance < min_explained_variance:
        return fallback(
            "MOTION_REFERENCE_NOT_INFORMATIVE",
            explained=explained_variance,
        )

    correction_fraction = (
        calculate_rms(estimated_noise)
        / max(calculate_rms(ecg_centered), 1e-12)
    )

    if correction_fraction > max_correction_fraction:
        return fallback(
            "CORRECTION_EXCEEDS_SAFETY_LIMIT",
            explained=explained_variance,
            correction=correction_fraction,
        )

    candidate = ecg - estimated_noise

    if not np.isfinite(candidate).all():
        return fallback("INVALID_COMPENSATED_SIGNAL")

    return {
        "output_ecg": candidate,
        "candidate_ecg": candidate.copy(),
        "accepted": True,
        "decision": "ACCEPT_CANDIDATE",
        "reason": "HEURISTIC_CHECKS_PASSED",
        "motion_rms": motion_rms,
        "explained_variance": float(explained_variance),
        "correction_fraction": float(correction_fraction),
        "method": "motion_reference_regression",
        "clinical_validated": False,
    }
