
"""
TRACE — Stage 12.2
Event-Local ECG Quality and Trust Analysis

Uses:
- Existing 300-second MIT-BIH replay
- Existing transient-event detector
- Existing ECG quality assessment
- Existing LightGBM artefact classifier
- Existing event trust router

Research-only software prototype.

Important:
- No synchronized physical IMU is available.
- Motion is NOT inferred from a zero-valued placeholder.
- Classifier outputs are experimental and may be wrong.
- No compensation is attempted without a motion reference.
- Event candidates are not diagnoses.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
)
from app.ecg_quality import analyze_ecg_quality
from app.artifact_inference import ArtifactInference
from app.event_trust_router import (
    route_event,
    LOW_CONFIDENCE_REVIEW,
)


WINDOW_SECONDS = 4.0
HALF_WINDOW_SAMPLES = int(FS * WINDOW_SECONDS / 2)


def _extract_label(prediction: Any) -> str:
    if not isinstance(prediction, dict):
        return "unknown"

    for key in (
        "predicted_class",
        "predicted_label",
        "artifact_label",
        "label",
        "class_name",
    ):
        value = prediction.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return "unknown"


def _event_time(event: dict) -> float:
    """
    Support common timestamp field names without
    silently interpreting sample indices as seconds.
    """
    for key in (
        "timestamp_seconds",
        "time_seconds",
        "event_time_seconds",
    ):
        if key in event:
            return float(event[key])

    raise ValueError(
        "Event has no recognized timestamp-in-seconds field. "
        f"Available keys: {sorted(event.keys())}"
    )


@lru_cache(maxsize=1)
def _classifier():
    return ArtifactInference()


def analyze_event_local(
    event: dict,
    ecg: np.ndarray,
) -> dict:
    """
    Analyze a four-second ECG window around one event.

    Missing physical motion evidence forces conservative
    review status. A synthetic zero motion vector is used
    only to satisfy the classifier input interface.
    """

    if not isinstance(event, dict):
        raise TypeError("event must be a dictionary")

    signal = np.asarray(ecg, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("ECG must be one-dimensional")

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains non-finite values")

    event_seconds = _event_time(event)
    event_sample = int(round(event_seconds * FS))

    start = event_sample - HALF_WINDOW_SAMPLES
    end = event_sample + HALF_WINDOW_SAMPLES

    if start < 0 or end > len(signal):
        result = route_event(
            event,
            artifact_label=None,
            quality_flags=None,
        )
        result["local_analysis"] = {
            "available": False,
            "reason": "INSUFFICIENT_EVENT_WINDOW",
            "motion_reference_available": False,
            "compensation_attempted": False,
        }
        return result

    window = signal[start:end]

    quality = analyze_ecg_quality(window, fs=FS)
    flags = list(quality.quality_flags)

    # Placeholder only. This is not measured motion.
    unavailable_motion = np.zeros_like(window)

    prediction = _classifier().predict(
        window,
        unavailable_motion,
    )
    label = _extract_label(prediction)

    # Route using available ECG evidence, but never
    # imply a physical motion measurement exists.
    routed = route_event(
        event=event,
        artifact_label=label,
        quality_flags=flags,
        compensation={
            "accepted": False,
            "decision": "NOT_ATTEMPTED",
            "reason": "MOTION_REFERENCE_UNAVAILABLE",
        },
        motion_suspected=None,
    )

    # The existing router does not model absent
    # physical motion evidence as a separate input.
    # Override to conservative review until Stage 12.3
    # introduces explicit evidence availability.
    routed["trust_status"] = LOW_CONFIDENCE_REVIEW

    reasons = list(routed["trust_reasons"])
    if "MOTION_REFERENCE_UNAVAILABLE" not in reasons:
        reasons.append("MOTION_REFERENCE_UNAVAILABLE")

    routed["trust_reasons"] = reasons

    routed["trust_evidence"][
        "motion_reference_available"
    ] = False

    routed["local_analysis"] = {
        "available": True,
        "window_start_seconds": start / FS,
        "window_end_seconds": end / FS,
        "sample_count": len(window),
        "artifact_label": label,
        "quality_flags": flags,
        "motion_reference_available": False,
        "motion_input_type": "ZERO_PLACEHOLDER_NOT_IMU",
        "compensation_attempted": False,
        "artifact_prediction_keys": (
            list(prediction.keys())
            if isinstance(prediction, dict)
            else []
        ),
        "clinical_validated": False,
    }

    return routed


def analyze_recording_events() -> dict:
    """
    Detect events from the complete evaluation segment,
    then independently analyze each event-local window.
    """

    ecg = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    events = detect_events(ecg)

    analyzed = [
        analyze_event_local(event, ecg)
        for event in events
    ]

    return {
        "source": "MIT_BIH_RECORD_100_REPLAY",
        "duration_seconds": len(ecg) / FS,
        "sample_rate_hz": FS,
        "event_count": len(analyzed),
        "events": analyzed,
        "motion_reference_available": False,
        "compensation_attempted": False,
        "clinical_validated": False,
    }
