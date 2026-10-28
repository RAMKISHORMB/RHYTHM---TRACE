
"""
TRACE — Stage 11.7
Motion Compensation Baseline Evaluation

Tests a conventional ECG bandpass filter against
synthetic continuous-motion artefacts.

This is a NON-ADAPTIVE baseline, not a validated
motion-compensation system.

The filter uses forward-backward processing,
which is offline and non-causal.

Research prototype only.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.continuous_motion_stress import (
    generate_corruption,
)
from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)


SCENARIOS = (
    "SUSTAINED_WALKING",
    "REPEATED_ELECTRODE_POPS",
)

STRENGTHS = (1.0, 2.0, 4.0)
SEED = 42

FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 35.0

START_SECONDS = 170.0
DURATION_SECONDS = 120.0


def rms(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError("Cannot calculate RMS of empty data.")

    return float(np.sqrt(np.mean(values ** 2)))


def apply_ecg_filter(ecg, fs=FS):
    """
    Apply a fourth-order Butterworth bandpass.

    Offline zero-phase filtering is used only
    for this research comparison.
    """

    ecg = np.asarray(ecg, dtype=np.float64)

    if ecg.ndim != 1 or len(ecg) < 100:
        raise ValueError("Invalid ECG input.")

    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains invalid values.")

    sos = butter(
        4,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, ecg)


def evaluate_compensation(
    scenario,
    strength,
    seed=SEED,
):
    """
    Compare raw and filtered corrupted ECG.

    All signal-error measurements use the known
    uncorrupted ECG available in simulation.
    """

    reference = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    observed, _ = generate_corruption(
        scenario=scenario,
        strength=strength,
        seed=seed,
    )

    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    filtered = apply_ecg_filter(observed)
    filtered_clean = apply_ecg_filter(reference)

    start = int(round(START_SECONDS * FS))
    end = int(
        round(
            (START_SECONDS + DURATION_SECONDS) * FS
        )
    )

    ref_region = reference[start:end]
    raw_region = observed[start:end]
    filtered_region = filtered[start:end]
    filtered_clean_region = filtered_clean[start:end]

    raw_error = rms(raw_region - ref_region)

    filtered_error = rms(
        filtered_region - ref_region
    )

    clean_filter_distortion = rms(
        filtered_clean_region - ref_region
    )

    # Isolates the corruption remaining after
    # applying the same filter to both signals.
    residual_artifact_error = rms(
        filtered_region - filtered_clean_region
    )

    improvement_percent = (
        100.0 * (raw_error - filtered_error) / raw_error
        if raw_error > 0
        else 0.0
    )

    reference_events = detect_events(reference)
    raw_events = detect_events(observed)
    filtered_events = detect_events(filtered)

    raw_comparison = match_events(
        reference_events,
        raw_events,
    )

    filtered_comparison = match_events(
        reference_events,
        filtered_events,
    )

    return {
        "scenario": scenario,
        "strength": strength,
        "seed": seed,
        "raw_error_rms": round(raw_error, 6),
        "filtered_error_rms": round(filtered_error, 6),
        "improvement_percent": round(
            improvement_percent, 3
        ),
        "clean_filter_distortion_rms": round(
            clean_filter_distortion, 6
        ),
        "residual_artifact_rms": round(
            residual_artifact_error, 6
        ),
        "reference_event_count": len(reference_events),
        "raw_event_count": len(raw_events),
        "filtered_event_count": len(filtered_events),
        "raw_agreement_f1": raw_comparison[
            "agreement_f1"
        ],
        "filtered_agreement_f1": filtered_comparison[
            "agreement_f1"
        ],
        "raw_missed_events": raw_comparison[
            "unmatched_reference_events"
        ],
        "filtered_missed_events": filtered_comparison[
            "unmatched_reference_events"
        ],
        "raw_extra_events": raw_comparison[
            "unmatched_observed_events"
        ],
        "filtered_extra_events": filtered_comparison[
            "unmatched_observed_events"
        ],
        "filter_type": "offline_butterworth_bandpass",
        "motion_adaptive": False,
        "clinical_validated": False,
    }
