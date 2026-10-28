
"""
TRACE — Stage 11.12
Event Trust Router Verification

Tests all four review states and precedence rules.
"""

from app.event_trust_router import (
    route_event,
    route_events,
    REVIEW_CANDIDATE,
    MOTION_AFFECTED_REVIEW,
    COMPENSATED_REVIEW,
    LOW_CONFIDENCE_REVIEW,
)


def sample_event():
    return {
        "event_type": "SHORT_RR_CANDIDATE",
        "timestamp_seconds": 185.532,
        "rr_interval_seconds": 0.52,
        "baseline_rr_seconds": 0.81,
    }


def check(name, expected, **kwargs):
    result = route_event(
        event=sample_event(),
        **kwargs,
    )

    actual = result["trust_status"]

    assert actual == expected, (
        f"{name}: expected {expected}, got {actual}"
    )

    assert result["requires_review"] is True

    assert (
        result["diagnostic_status"]
        == "UNCONFIRMED_EVENT_CANDIDATE"
    )

    assert result["clinical_validated"] is False

    print(
        f"PASS | {name:<32} | {actual}"
    )

    return result


def main():
    print("=" * 76)
    print("TRACE STAGE 11.12 — EVENT TRUST ROUTER")
    print("=" * 76)

    check(
        "Clean candidate",
        REVIEW_CANDIDATE,
        artifact_label="clean",
        quality_flags=[],
    )

    check(
        "Motion suspected",
        MOTION_AFFECTED_REVIEW,
        artifact_label="walking",
        quality_flags=[],
    )

    check(
        "Accepted compensation",
        COMPENSATED_REVIEW,
        artifact_label="walking",
        quality_flags=[],
        compensation={
            "accepted": True,
            "decision": "ACCEPT_CORRECTION",
            "reason": "HEURISTIC_CHECKS_PASSED",
        },
    )

    check(
        "Motion with fallback",
        LOW_CONFIDENCE_REVIEW,
        artifact_label="walking",
        quality_flags=[],
        compensation={
            "accepted": False,
            "decision": "PRESERVE_ORIGINAL",
            "reason": "MOTION_REFERENCE_NOT_INFORMATIVE",
        },
    )

    check(
        "Severe electrode pop",
        LOW_CONFIDENCE_REVIEW,
        artifact_label="electrode_pop",
        quality_flags=[],
    )

    check(
        "Lead dropout",
        LOW_CONFIDENCE_REVIEW,
        artifact_label="lead_dropout",
        quality_flags=["lead_dropout"],
    )

    check(
        "Missing evidence",
        LOW_CONFIDENCE_REVIEW,
    )

    check(
        "Unknown artefact",
        LOW_CONFIDENCE_REVIEW,
        artifact_label="unknown",
        quality_flags=[],
    )

    original = sample_event()

    routed = route_events(
        [original, sample_event()],
        artifact_label="clean",
        quality_flags=[],
    )

    assert len(routed) == 2
    assert all(
        event["trust_status"] == REVIEW_CANDIDATE
        for event in routed
    )

    assert "trust_status" not in original

    print(
        "PASS | Collection routing and immutability"
    )

    print()
    print("=" * 76)
    print("PASS: All Stage 11.12 routing tests completed.")
    print("=" * 76)


if __name__ == "__main__":
    main()
