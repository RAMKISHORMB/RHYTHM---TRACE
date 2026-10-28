
"""
TRACE — Stage 11.12
Explainable Event Trust Router

Purpose:
Route transient cardiac event candidates for review
using signal quality, artefact and compensation evidence.

This is NOT a diagnostic confidence model.
All outputs remain review candidates.

Research prototype only.
"""

from typing import Any


REVIEW_CANDIDATE = "REVIEW_CANDIDATE"
MOTION_AFFECTED_REVIEW = "MOTION_AFFECTED_REVIEW"
COMPENSATED_REVIEW = "COMPENSATED_REVIEW"
LOW_CONFIDENCE_REVIEW = "LOW_CONFIDENCE_REVIEW"


MOTION_LABELS = {
    "walking",
    "baseline_wander",
    "electrode_pop",
    "lead_dropout",
    "motion",
    "sustained_walking",
    "repeated_electrode_pops",
}

SEVERE_LABELS = {
    "electrode_pop",
    "lead_dropout",
    "repeated_electrode_pops",
}

SEVERE_QUALITY_FLAGS = {
    "lead_dropout",
    "flatline",
    "signal_saturation",
    "severe_noise",
    "non_finite_signal",
}


def _normalize_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")


def _normalize_flags(flags: Any) -> list[str]:
    if flags is None:
        return []

    if isinstance(flags, str):
        flags = [flags]

    if not isinstance(flags, (list, tuple, set)):
        return ["unrecognized_quality_flags"]

    return sorted(
        {
            _normalize_label(flag)
            for flag in flags
            if flag is not None
        }
    )


def route_event(
    event: dict,
    artifact_label: str | None = None,
    quality_flags: list[str] | None = None,
    compensation: dict | None = None,
    motion_suspected: bool | None = None,
) -> dict:
    """
    Route one event candidate.

    Conservative precedence:
    1. Missing or severe evidence -> low confidence
    2. Rejected compensation during suspected motion
       -> low confidence
    3. Accepted compensation -> compensated review
    4. Motion evidence -> motion-affected review
    5. Otherwise -> ordinary review

    Missing evidence is not interpreted as clean ECG.
    """

    if not isinstance(event, dict):
        raise TypeError("event must be a dictionary")

    routed = event.copy()

    label = _normalize_label(artifact_label)
    flags = _normalize_flags(quality_flags)

    if compensation is not None and not isinstance(
        compensation, dict
    ):
        raise TypeError("compensation must be a dictionary")

    if motion_suspected is not None and not isinstance(
        motion_suspected, bool
    ):
        raise TypeError("motion_suspected must be bool or None")

    compensation = compensation or {}

    accepted = compensation.get("accepted", False)

    if not isinstance(accepted, bool):
        raise TypeError("compensation.accepted must be boolean")

    compensation_decision = str(
        compensation.get("decision", "NOT_ATTEMPTED")
    )

    compensation_reason = str(
        compensation.get("reason", "NOT_PROVIDED")
    )

    has_artifact_evidence = artifact_label is not None
    has_quality_evidence = quality_flags is not None

    evidence_missing = not (
        has_artifact_evidence and has_quality_evidence
    )

    severe_artifact = label in SEVERE_LABELS

    severe_quality = bool(
        set(flags) & SEVERE_QUALITY_FLAGS
    )

    motion_detected = (
        label in MOTION_LABELS
        if motion_suspected is None
        else motion_suspected
    )

    reasons = []

    if evidence_missing:
        status = LOW_CONFIDENCE_REVIEW
        reasons.append("INCOMPLETE_SIGNAL_EVIDENCE")

    elif severe_artifact or severe_quality:
        status = LOW_CONFIDENCE_REVIEW
        reasons.append("SEVERE_SIGNAL_INTERFERENCE")

    elif (
        motion_detected
        and compensation_decision == "PRESERVE_ORIGINAL"
    ):
        status = LOW_CONFIDENCE_REVIEW
        reasons.append("MOTION_UNRESOLVED_AFTER_FALLBACK")

    elif accepted:
        status = COMPENSATED_REVIEW
        reasons.append("MOTION_CORRECTION_ACCEPTED")

    elif motion_detected:
        status = MOTION_AFFECTED_REVIEW
        reasons.append("MOTION_INTERFERENCE_SUSPECTED")

    elif label == "clean" and not flags:
        status = REVIEW_CANDIDATE
        reasons.append("NO_KNOWN_INTERFERENCE")

    else:
        status = LOW_CONFIDENCE_REVIEW
        reasons.append("UNCLASSIFIED_SIGNAL_CONDITION")

    routed["trust_status"] = status

    routed["trust_reasons"] = reasons

    routed["trust_evidence"] = {
        "artifact_label": label,
        "quality_flags": flags,
        "motion_suspected": motion_detected,
        "compensation_accepted": accepted,
        "compensation_decision": compensation_decision,
        "compensation_reason": compensation_reason,
    }

    routed["diagnostic_status"] = (
        "UNCONFIRMED_EVENT_CANDIDATE"
    )

    routed["requires_review"] = True

    routed["clinical_validated"] = False

    return routed


def route_events(
    events: list[dict],
    artifact_label: str | None = None,
    quality_flags: list[str] | None = None,
    compensation: dict | None = None,
    motion_suspected: bool | None = None,
) -> list[dict]:
    """
    Route a collection of event candidates.

    Current implementation applies the same evidence
    to every event in the supplied collection.

    For per-event routing, pass event-local evidence
    separately in a future integration stage.
    """

    if not isinstance(events, list):
        raise TypeError("events must be a list")

    return [
        route_event(
            event=event,
            artifact_label=artifact_label,
            quality_flags=quality_flags,
            compensation=compensation,
            motion_suspected=motion_suspected,
        )
        for event in events
    ]
