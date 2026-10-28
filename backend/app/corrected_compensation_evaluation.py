
"""
TRACE — Stage 11.7.2
Corrected ECG Compensation Evaluation

Separates:
1. Artefact suppression
2. Clean ECG distortion
3. Event detection agreement

Uses an offline, non-adaptive bandpass filter.

Research only. Not clinically validated.
"""

import numpy as np

from app.continuous_motion_stress import (
    generate_corruption,
)
from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)
from app.motion_compensation_baseline import (
    apply_ecg_filter,
)


SCENARIOS = (
    "SUSTAINED_WALKING",
    "REPEATED_ELECTRODE_POPS",
)

STRENGTHS = (1.0, 2.0, 4.0)

SEED = 42

START_SECONDS = 170.0
DURATION_SECONDS = 120.0


def rms(signal):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        raise ValueError("Empty signal.")

    return float(
        np.sqrt(np.mean(signal ** 2))
    )


def center(signal):
    signal = np.asarray(signal, dtype=np.float64)
    return signal - np.median(signal)


def evaluate_corrected_compensation(
    scenario,
    strength,
    seed=SEED,
):
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

    filtered_reference = apply_ecg_filter(
        reference,
        fs=FS,
    )

    filtered_observed = apply_ecg_filter(
        observed,
        fs=FS,
    )

    start = int(round(START_SECONDS * FS))
    end = int(
        round(
            (START_SECONDS + DURATION_SECONDS) * FS
        )
    )

    ref = reference[start:end]
    obs = observed[start:end]

    filtered_ref = filtered_reference[start:end]
    filtered_obs = filtered_observed[start:end]

    # ---------------------------------------------
    # 1. RAW ARTEFACT ERROR
    # ---------------------------------------------

    raw_artifact_rms = rms(obs - ref)

    # ---------------------------------------------
    # 2. RESIDUAL ARTEFACT AFTER FILTERING
    # ---------------------------------------------
    # Compare filtered observed ECG against
    # filtered clean ECG, not raw clean ECG.
    # This isolates the additive corruption
    # remaining after identical filtering.
    # ---------------------------------------------

    residual_artifact_rms = rms(
        filtered_obs - filtered_ref
    )

    artifact_reduction_percent = (
        100.0
        * (
            raw_artifact_rms
            - residual_artifact_rms
        )
        / raw_artifact_rms
        if raw_artifact_rms > 0
        else 0.0
    )

    # ---------------------------------------------
    # 3. CLEAN ECG WAVEFORM DISTORTION
    # ---------------------------------------------
    # Median centering removes constant offset
    # but does not remove morphology changes.
    # ---------------------------------------------

    clean_distortion_rms = rms(
        center(filtered_ref) - center(ref)
    )

    clean_signal_rms = rms(center(ref))

    relative_clean_distortion_percent = (
        100.0
        * clean_distortion_rms
        / clean_signal_rms
        if clean_signal_rms > 0
        else 0.0
    )

    # ---------------------------------------------
    # 4. EVENT DETECTION
    # ---------------------------------------------

    reference_events = detect_events(reference)
    raw_events = detect_events(observed)
    filtered_events = detect_events(
        filtered_observed
    )

    raw_match = match_events(
        reference_events,
        raw_events,
    )

    filtered_match = match_events(
        reference_events,
        filtered_events,
    )

    return {
        "scenario": scenario,
        "strength": strength,
        "seed": seed,

        "raw_artifact_rms": round(
            raw_artifact_rms, 6
        ),

        "residual_artifact_rms": round(
            residual_artifact_rms, 6
        ),

        "artifact_reduction_percent": round(
            artifact_reduction_percent, 3
        ),

        "clean_distortion_rms": round(
            clean_distortion_rms, 6
        ),

        "relative_clean_distortion_percent": round(
            relative_clean_distortion_percent, 3
        ),

        "reference_event_count": len(
            reference_events
        ),

        "raw_event_count": len(raw_events),

        "filtered_event_count": len(
            filtered_events
        ),

        "raw_agreement_f1": raw_match[
            "agreement_f1"
        ],

        "filtered_agreement_f1": filtered_match[
            "agreement_f1"
        ],

        "raw_missed_events": raw_match[
            "unmatched_reference_events"
        ],

        "filtered_missed_events": filtered_match[
            "unmatched_reference_events"
        ],

        "raw_extra_events": raw_match[
            "unmatched_observed_events"
        ],

        "filtered_extra_events": filtered_match[
            "unmatched_observed_events"
        ],

        "evaluation_type": (
            "filtered_reference_artifact_comparison"
        ),

        "filter_type": (
            "offline_nonadaptive_bandpass"
        ),

        "clinical_validated": False,
    }
