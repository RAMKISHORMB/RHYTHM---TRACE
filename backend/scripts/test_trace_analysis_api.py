
"""
TRACE — Stage 12.3
Unified Analysis API Verification
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def main():
    print("=" * 72)
    print("TRACE STAGE 12.3 — API VERIFICATION")
    print("=" * 72)

    response = client.get(
        "/api/trace/unified-window",
        params={
            "scenario": "REST",
            "start_seconds": 180.0,
            "seed": 42,
        },
    )

    assert response.status_code == 200, (
        response.text
    )

    window = response.json()

    assert window["sample_count"] == 1000
    assert window["scenario"] == "REST"
    assert len(window["output_ecg"]) == 1000

    print("PASS | Unified REST window endpoint")

    response = client.get(
        "/api/trace/unified-window",
        params={
            "scenario": "WALKING",
            "seed": 42,
        },
    )

    assert response.status_code == 200, (
        response.text
    )

    walking = response.json()

    assert walking["scenario"] == "WALKING"

    print("PASS | Unified WALKING endpoint")

    response = client.get(
        "/api/trace/unified-window",
        params={
            "scenario": "INVALID",
        },
    )

    assert response.status_code == 422

    print("PASS | Invalid scenario rejected")

    response = client.get(
        "/api/trace/event-analysis"
    )

    assert response.status_code == 200, (
        response.text
    )

    report = response.json()

    assert report["event_count"] == 3
    assert len(report["events"]) == 3

    for event in report["events"]:
        assert (
            event["trust_status"]
            == "LOW_CONFIDENCE_REVIEW"
        )

        assert (
            "MOTION_REFERENCE_UNAVAILABLE"
            in event["trust_reasons"]
        )

        assert (
            "NO_KNOWN_INTERFERENCE"
            not in event["trust_reasons"]
        )

        assert event["requires_review"] is True

    print("PASS | Event-local analysis endpoint")
    print("PASS | Conservative trust evidence")

    print()
    print("Detected events:", report["event_count"])

    for event in report["events"]:
        print(
            "Event:",
            event.get("event_type"),
            "| Time:",
            event.get("timestamp_seconds"),
            "| Status:",
            event["trust_status"],
        )

    print()
    print("=" * 72)
    print("PASS: All Stage 12.3 API tests completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
