
"""
TRACE — Virtual Wearable Event Pipeline
MEDAITHON'26 | Cardiology PS-001
Stage 11.3

Integrates:
    1. Recorded MIT-BIH ECG
    2. Reference-based transient-event detection
    3. Virtual wearable scenarios
    4. LightGBM artefact classification
    5. ECG quality assessment
    6. Event preservation and review status

IMPORTANT:
    Cardiac event detection in this stage uses
    the uncorrupted recorded ECG.

    The simulated wearable context is attached
    separately for experimental evaluation.

    This does not demonstrate detection accuracy
    on motion-corrupted ECG.

Research prototype only.
Not clinically validated.
"""

from functools import lru_cache
from typing import Any

from app.virtual_wearable import (
    FS,
    load_recorded_ecg,
)

from app.virtual_wearable_analysis import (
    analyze_virtual_wearable,
)

from app.transient_event_detector import (
    detect_transient_events,
)


# ============================================================
# CONFIGURATION
# ============================================================

RECORDING_ID = "MITBIH-100"

ANALYSIS_DURATION_SECONDS = 300

EVENT_MATCH_TOLERANCE_SECONDS = 0.1

SUPPORTED_SCENARIOS = (
    "REST",
    "WALKING",
    "ELECTRODE_POP",
)


# ============================================================
# LOAD AND DETECT REFERENCE EVENTS
# ============================================================

@lru_cache(maxsize=1)
def get_reference_events():
    """
    Detect transient-event candidates in the
    first 300 seconds of recorded MIT-BIH ECG.

    The existing detector is reused unchanged.
    """

    ecg = load_recorded_ecg()

    required_samples = (
        ANALYSIS_DURATION_SECONDS * FS
    )

    if len(ecg) < required_samples:
        raise ValueError(
            "Recorded ECG is shorter than "
            "the requested analysis duration."
        )

    segment = ecg[:required_samples]

    detection = detect_transient_events(
        segment,
        fs=FS,
    )

    return detection


# ============================================================
# NORMALIZE EVENT DATA
# ============================================================

def normalize_event(event: Any) -> dict:
    """
    Convert an event to a dictionary.

    Supports the existing detector's dictionary
    events and object-based event structures.
    """

    if isinstance(event, dict):
        return dict(event)

    if hasattr(event, "to_dict"):
        result = event.to_dict()

        if isinstance(result, dict):
            return result

    fields = (
        "event_type",
        "timestamp_seconds",
        "rr_interval_seconds",
        "baseline_rr_seconds",
        "ratio_to_baseline",
    )

    return {
        field: getattr(event, field)
        for field in fields
    }


# ============================================================
# EXTRACT EVENT COLLECTION
# ============================================================

def extract_events(detection: Any) -> list[dict]:
    """
    Extract detected events from the existing
    transient-event detector result.
    """

    if isinstance(detection, dict):
        raw_events = detection.get("events", [])

    elif hasattr(detection, "events"):
        raw_events = detection.events

    else:
        raise TypeError(
            "Unsupported transient-event "
            "detector result format."
        )

    events = [
        normalize_event(event)
        for event in raw_events
    ]

    return events


# ============================================================
# FIND NEAREST REFERENCE EVENT
# ============================================================

def find_reference_event(
    event_time_seconds: float,
) -> dict:
    """
    Find the detected event nearest the
    requested timestamp.
    """

    detection = get_reference_events()

    events = extract_events(detection)

    if not events:
        raise ValueError(
            "No transient-event candidates found."
        )

    nearest_event = min(
        events,
        key=lambda event: abs(
            float(event["timestamp_seconds"])
            - event_time_seconds
        ),
    )

    difference = abs(
        float(
            nearest_event["timestamp_seconds"]
        )
        - event_time_seconds
    )

    if difference > EVENT_MATCH_TOLERANCE_SECONDS:
        raise ValueError(
            "No reference event found within "
            f"{EVENT_MATCH_TOLERANCE_SECONDS} "
            "seconds of requested timestamp."
        )

    return nearest_event


# ============================================================
# UNIFIED EVENT + WEARABLE ANALYSIS
# ============================================================

def analyze_event_scenario(
    scenario: str = "REST",
    event_time_seconds: float = 185.532,
    seed: int = 42,
) -> dict:
    """
    Analyze a reference-detected cardiac-event
    candidate with simulated wearable context.

    The candidate is always preserved.

    Artefact and ECG-quality results determine
    review context, not event deletion.
    """

    scenario = scenario.upper().strip()

    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}. "
            f"Supported: {SUPPORTED_SCENARIOS}"
        )

    # --------------------------------------------------------
    # 1. FIND REFERENCE-DETECTED EVENT
    # --------------------------------------------------------

    event = find_reference_event(
        event_time_seconds
    )

    event_timestamp = float(
        event["timestamp_seconds"]
    )

    # --------------------------------------------------------
    # 2. GENERATE FOUR-SECOND WEARABLE CONTEXT
    # --------------------------------------------------------

    window_start_seconds = (
        event_timestamp - 2.0
    )

    wearable = analyze_virtual_wearable(
        scenario=scenario,
        start_seconds=window_start_seconds,
        seed=seed,
    )

    # --------------------------------------------------------
    # 3. READ ARTEFACT CLASSIFICATION
    # --------------------------------------------------------

    prediction = wearable[
        "artifact_prediction"
    ]

    predicted_class = prediction.get(
        "predicted_class",
        "unknown",
    )

    model_probability = prediction.get(
        "model_probability"
    )

    # --------------------------------------------------------
    # 4. READ ECG QUALITY CONTEXT
    # --------------------------------------------------------

    quality_flags = list(
        wearable.get("quality_flags", [])
    )

    # --------------------------------------------------------
    # 5. ASSIGN REVIEW STATUS
    # --------------------------------------------------------

    artefact_suspected = (
        predicted_class != "clean"
    )

    quality_warning = (
        len(quality_flags) > 0
    )

    if artefact_suspected or quality_warning:
        review_status = "REVIEW_SIGNAL_QUALITY"

    else:
        review_status = "REVIEW_EVENT"

    # The event must never be discarded merely
    # because an artefact is suspected.

    event_preserved = True

    # --------------------------------------------------------
    # 6. CREATE EXPLANATION
    # --------------------------------------------------------

    if review_status == "REVIEW_SIGNAL_QUALITY":
        explanation = (
            "A transient cardiac-event candidate "
            "was identified in the recorded reference "
            "ECG. The simulated wearable context "
            "contains a suspected artefact or ECG "
            "quality warning. The candidate is "
            "preserved for further review."
        )

    else:
        explanation = (
            "A transient cardiac-event candidate "
            "was identified in the recorded reference "
            "ECG. The current experimental wearable "
            "analysis produced no artefact or quality "
            "warning. The event still requires review."
        )

    # --------------------------------------------------------
    # 7. RETURN UNIFIED RESULT
    # --------------------------------------------------------

    return {
        "recording_id": RECORDING_ID,

        "event_type": event["event_type"],

        "event_timestamp_seconds": (
            event_timestamp
        ),

        "rr_interval_seconds": (
            event["rr_interval_seconds"]
        ),

        "baseline_rr_seconds": (
            event["baseline_rr_seconds"]
        ),

        "ratio_to_baseline": (
            event["ratio_to_baseline"]
        ),

        "scenario": scenario,

        "window_start_seconds": (
            wearable["start_seconds"]
        ),

        "window_end_seconds": (
            wearable["end_seconds"]
        ),

        "sampling_rate": (
            wearable["sampling_rate"]
        ),

        "motion_rms": (
            wearable["motion_rms"]
        ),

        "motion_source": (
            wearable["motion_source"]
        ),

        "motion_is_simulated": True,

        "artifact_prediction": prediction,

        "predicted_artifact_class": (
            predicted_class
        ),

        "artifact_model_probability": (
            model_probability
        ),

        "quality_flags": quality_flags,

        "review_status": review_status,

        "event_preserved": event_preserved,

        "event_detection_source": (
            "recorded_reference_ecg"
        ),

        "event_detected_from_corrupted_ecg": False,

        "explanation": explanation,

        "requires_review": True,

        "research_only": True,

        "clinical_validated": False,

        "live_patient_monitoring": False,
    }
