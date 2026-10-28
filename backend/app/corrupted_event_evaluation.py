
"""
TRACE — Corrupted ECG Event Evaluation
MEDAITHON'26 | Cardiology PS-001
Stage 11.4

Compares transient-event detection on:
1. Original recorded ECG
2. ECG with simulated wearable artefacts

Research evaluation only.
No clinical validation or real wearable motion data.
"""

from functools import lru_cache

import numpy as np

from app.virtual_wearable import (
    FS,
    load_recorded_ecg,
    generate_wearable_window,
)
from app.transient_event_detector import (
    detect_transient_events,
)
from app.virtual_event_pipeline import (
    extract_events,
)


DURATION_SECONDS = 300
TARGET_EVENT_SECONDS = 185.532
WINDOW_SECONDS = 4
MATCH_TOLERANCE_SECONDS = 0.15

SCENARIOS = (
    "REST",
    "WALKING",
    "ELECTRODE_POP",
)


@lru_cache(maxsize=1)
def load_evaluation_ecg() -> np.ndarray:
    """Load the first 300 seconds of recorded ECG."""

    ecg = np.asarray(
        load_recorded_ecg(),
        dtype=np.float64,
    )

    required = DURATION_SECONDS * FS

    if len(ecg) < required:
        raise ValueError(
            "Insufficient ECG for evaluation."
        )

    segment = ecg[:required].copy()
    segment.setflags(write=False)

    return segment


def detect_events(ecg: np.ndarray) -> list[dict]:
    """Run the existing detector unchanged."""

    result = detect_transient_events(
        ecg,
        fs=FS,
    )

    return extract_events(result)


def match_events(
    reference_events: list[dict],
    observed_events: list[dict],
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
) -> dict:
    """
    One-to-one event matching by timestamp.

    Each observed event can match at most one
    reference event of the same event type.
    """

    candidates = []

    for ref_index, reference in enumerate(
        reference_events
    ):
        for obs_index, observed in enumerate(
            observed_events
        ):
            if (
                reference["event_type"]
                != observed["event_type"]
            ):
                continue

            error = abs(
                float(reference["timestamp_seconds"])
                - float(observed["timestamp_seconds"])
            )

            if error <= tolerance_seconds:
                candidates.append(
                    (error, ref_index, obs_index)
                )

    candidates.sort()

    matched_reference = set()
    matched_observed = set()
    matches = []

    for error, ref_index, obs_index in candidates:
        if (
            ref_index in matched_reference
            or obs_index in matched_observed
        ):
            continue

        matched_reference.add(ref_index)
        matched_observed.add(obs_index)

        matches.append({
            "reference_timestamp_seconds": float(
                reference_events[ref_index][
                    "timestamp_seconds"
                ]
            ),
            "observed_timestamp_seconds": float(
                observed_events[obs_index][
                    "timestamp_seconds"
                ]
            ),
            "event_type": reference_events[
                ref_index
            ]["event_type"],
            "timing_error_ms": round(
                error * 1000,
                3,
            ),
        })

    tp = len(matches)
    fn = len(reference_events) - tp
    fp = len(observed_events) - tp

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "matched_events": tp,
        "unmatched_reference_events": fn,
        "unmatched_observed_events": fp,
        "agreement_precision": round(
            precision, 4
        ),
        "agreement_recall": round(
            recall, 4
        ),
        "agreement_f1": round(
            f1, 4
        ),
        "matches": matches,
    }


def evaluate_corrupted_scenario(
    scenario: str = "REST",
    target_event_seconds: float = (
        TARGET_EVENT_SECONDS
    ),
    seed: int = 42,
) -> dict:
    """
    Inject one simulated artefact window into
    a 300-second recorded ECG and compare
    detected candidates.

    Reference events are algorithm outputs,
    not independent clinical ground truth.
    """

    scenario = scenario.upper().strip()

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    reference_ecg = load_evaluation_ecg()

    observed_ecg = reference_ecg.copy()

    # Center the four-second window on
    # the selected candidate.
    start_seconds = (
        target_event_seconds
        - WINDOW_SECONDS / 2
    )

    window = generate_wearable_window(
        scenario=scenario,
        start_seconds=start_seconds,
        seed=seed,
    )

    start_sample = int(
        round(start_seconds * FS)
    )

    observed_window = np.asarray(
        window.ecg_observed,
        dtype=np.float64,
    )

    end_sample = (
        start_sample + len(observed_window)
    )

    if (
        start_sample < 0
        or end_sample > len(observed_ecg)
    ):
        raise ValueError(
            "Corruption window outside recording."
        )

    # Validate that the window reference aligns
    # with the evaluation recording.
    reference_window = np.asarray(
        window.ecg_reference,
        dtype=np.float64,
    )

    original_window = observed_ecg[
        start_sample:end_sample
    ]

    if not np.allclose(
        original_window,
        reference_window,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError(
            "Virtual wearable window does not "
            "align with reference ECG."
        )

    observed_ecg[
        start_sample:end_sample
    ] = observed_window

    reference_events = detect_events(
        reference_ecg
    )

    observed_events = detect_events(
        observed_ecg
    )

    comparison = match_events(
        reference_events,
        observed_events,
    )

    target_reference = [
        event
        for event in reference_events
        if abs(
            float(event["timestamp_seconds"])
            - target_event_seconds
        ) <= MATCH_TOLERANCE_SECONDS
    ]

    if not target_reference:
        raise ValueError(
            "Target reference event not found."
        )

    target_detected = any(
        abs(
            match["reference_timestamp_seconds"]
            - target_event_seconds
        ) <= MATCH_TOLERANCE_SECONDS
        for match in comparison["matches"]
    )

    corruption_rms = float(
        np.sqrt(
            np.mean(
                (
                    observed_window
                    - reference_window
                ) ** 2
            )
        )
    )

    return {
        "scenario": scenario,
        "recording_id": "MITBIH-100",
        "duration_seconds": DURATION_SECONDS,
        "sampling_rate": FS,
        "corrupted_window_start_seconds": (
            start_seconds
        ),
        "corrupted_window_end_seconds": (
            start_seconds + WINDOW_SECONDS
        ),
        "reference_corruption_rms": (
            corruption_rms
        ),
        "reference_event_count": len(
            reference_events
        ),
        "observed_event_count": len(
            observed_events
        ),
        "target_event_seconds": (
            target_event_seconds
        ),
        "target_event_detected": (
            target_detected
        ),
        "comparison": comparison,
        "reference_events": reference_events,
        "observed_events": observed_events,
        "event_detection_source": (
            "observed_corrupted_ecg"
        ),
        "motion_source": (
            "synthetic_single_channel_proxy"
        ),
        "research_only": True,
        "clinical_validated": False,
    }
