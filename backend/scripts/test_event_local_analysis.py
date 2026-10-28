
"""
TRACE — Stage 12.2
Event-Local Analysis Verification
"""

import json
import numpy as np

from app.event_local_analysis import (
    analyze_recording_events,
    analyze_event_local,
)
from app.corrupted_event_evaluation import (
    load_evaluation_ecg,
)
from app.event_trust_router import (
    LOW_CONFIDENCE_REVIEW,
)


def main():
    print("=" * 72)
    print("TRACE STAGE 12.2 — EVENT-LOCAL TRUST ANALYSIS")
    print("=" * 72)

    report = analyze_recording_events()

    assert report["sample_rate_hz"] == 250
    assert report["event_count"] == len(report["events"])
    assert report["motion_reference_available"] is False
    assert report["compensation_attempted"] is False

    print("Recording duration:", report["duration_seconds"])
    print("Detected candidates:", report["event_count"])

    for index, event in enumerate(report["events"], start=1):
        assert event["trust_status"] == LOW_CONFIDENCE_REVIEW
        assert event["requires_review"] is True
        assert event["clinical_validated"] is False
        assert (
            "MOTION_REFERENCE_UNAVAILABLE"
            in event["trust_reasons"]
        )

        local = event["local_analysis"]

        assert local["available"] is True
        assert local["sample_count"] == 1000
        assert local["motion_reference_available"] is False
        assert local["compensation_attempted"] is False

        print()
        print("-" * 72)
        print("Event:", index)
        print("Event type:", event.get("event_type", "unknown"))
        print("Event time:", event.get("timestamp_seconds", "unknown"))
        print("Artefact label:", local["artifact_label"])
        print("Quality flags:", local["quality_flags"])
        print("Trust status:", event["trust_status"])
        print("Trust reasons:", event["trust_reasons"])

    # Explicit boundary test for an event at the
    # beginning of the recording.
    ecg = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    boundary = analyze_event_local(
        {
            "event_type": "TEST_BOUNDARY",
            "timestamp_seconds": 0.1,
        },
        ecg,
    )

    assert (
        boundary["local_analysis"]["available"]
        is False
    )

    assert (
        boundary["trust_status"]
        == LOW_CONFIDENCE_REVIEW
    )

    json.dumps(report, allow_nan=False)

    print()
    print("Boundary window test: PASS")
    print("JSON serialization: PASS")
    print("=" * 72)
    print("PASS: Stage 12.2 event-local analysis verified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
