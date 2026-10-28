
"""
TRACE — Stage 11.5
Transient Cardiac Event Stress Testing

24 controlled experiments:
    2 artefact types
    4 corruption strengths
    3 reference event positions

Compares event detection on corrupted ECG
against the detector's clean-ECG outputs.

Research only. Not clinically validated.
"""

from itertools import product

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    MATCH_TOLERANCE_SECONDS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)
from app.virtual_wearable import generate_wearable_window


SCENARIOS = ("WALKING", "ELECTRODE_POP")

STRENGTHS = (0.5, 1.0, 2.0, 4.0)

EVENT_TIMES = (
    185.532,
    208.292,
    276.608,
)

WINDOW_SECONDS = 4.0

SEED = 42


def build_corrupted_signal(
    scenario: str,
    strength: float,
    event_time_seconds: float,
    seed: int = SEED,
):
    """
    Apply scaled simulated artefact to a
    four-second region of recorded ECG.

    Strength 1.0 reproduces the original
    virtual wearable artefact amplitude.

    The original ECG is never modified.
    """

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    if not np.isfinite(strength) or strength < 0:
        raise ValueError(
            "Strength must be finite and nonnegative."
        )

    reference_ecg = load_evaluation_ecg()

    corrupted_ecg = reference_ecg.copy()

    start_seconds = (
        event_time_seconds - WINDOW_SECONDS / 2
    )

    window = generate_wearable_window(
        scenario=scenario,
        start_seconds=start_seconds,
        seed=seed,
    )

    reference_window = np.asarray(
        window.ecg_reference,
        dtype=np.float64,
    )

    observed_window = np.asarray(
        window.ecg_observed,
        dtype=np.float64,
    )

    start_sample = int(round(start_seconds * FS))

    end_sample = start_sample + len(
        reference_window
    )

    if start_sample < 0 or end_sample > len(
        corrupted_ecg
    ):
        raise ValueError(
            "Corruption window outside recording."
        )

    if not np.allclose(
        corrupted_ecg[start_sample:end_sample],
        reference_window,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError(
            "Virtual wearable ECG alignment failed."
        )

    artefact = observed_window - reference_window

    scaled_artefact = strength * artefact

    corrupted_ecg[start_sample:end_sample] = (
        reference_window + scaled_artefact
    )

    corruption_rms = float(
        np.sqrt(np.mean(scaled_artefact ** 2))
    )

    return corrupted_ecg, corruption_rms


def run_stress_test() -> list[dict]:
    """
    Run all 24 experiments.

    A failed event detection is recorded as
    a benchmark result, not a test exception.
    """

    reference_ecg = load_evaluation_ecg()

    reference_events = detect_events(
        reference_ecg
    )

    if not reference_events:
        raise RuntimeError(
            "No reference event candidates found."
        )

    results = []

    experiment_id = 0

    for scenario, strength, event_time in product(
        SCENARIOS,
        STRENGTHS,
        EVENT_TIMES,
    ):
        experiment_id += 1

        corrupted_ecg, corruption_rms = (
            build_corrupted_signal(
                scenario=scenario,
                strength=strength,
                event_time_seconds=event_time,
            )
        )

        observed_events = detect_events(
            corrupted_ecg
        )

        comparison = match_events(
            reference_events,
            observed_events,
        )

        target_detected = any(
            abs(
                match["reference_timestamp_seconds"]
                - event_time
            ) <= MATCH_TOLERANCE_SECONDS
            for match in comparison["matches"]
        )

        target_matches = [
            match
            for match in comparison["matches"]
            if abs(
                match["reference_timestamp_seconds"]
                - event_time
            ) <= MATCH_TOLERANCE_SECONDS
        ]

        target_timing_error_ms = (
            target_matches[0]["timing_error_ms"]
            if target_matches
            else None
        )

        results.append({
            "experiment_id": experiment_id,
            "scenario": scenario,
            "strength": strength,
            "target_event_seconds": event_time,
            "corruption_rms": round(
                corruption_rms, 6
            ),
            "reference_event_count": len(
                reference_events
            ),
            "observed_event_count": len(
                observed_events
            ),
            "matched_events": comparison[
                "matched_events"
            ],
            "missed_reference_events": comparison[
                "unmatched_reference_events"
            ],
            "extra_observed_events": comparison[
                "unmatched_observed_events"
            ],
            "agreement_precision": comparison[
                "agreement_precision"
            ],
            "agreement_recall": comparison[
                "agreement_recall"
            ],
            "agreement_f1": comparison[
                "agreement_f1"
            ],
            "target_event_detected": (
                target_detected
            ),
            "target_timing_error_ms": (
                target_timing_error_ms
            ),
            "motion_source": (
                "synthetic_single_channel_proxy"
            ),
            "clinical_validated": False,
        })

    return results
