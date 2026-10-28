
"""
TRACE — Stage 11.3 Verification

Tests reference-detected transient-event
preservation across virtual wearable scenarios.

Does not measure detection accuracy on
motion-corrupted ECG.
"""

import math

from app.virtual_event_pipeline import (
    analyze_event_scenario,
)


SCENARIOS = (
    "REST",
    "WALKING",
    "ELECTRODE_POP",
)

TARGET_EVENT_SECONDS = 185.532


def main():

    print("=" * 60)
    print("TRACE — STAGE 11.3")
    print("VIRTUAL WEARABLE EVENT INTEGRATION")
    print("=" * 60)

    results = []

    for scenario in SCENARIOS:

        result = analyze_event_scenario(
            scenario=scenario,
            event_time_seconds=TARGET_EVENT_SECONDS,
            seed=42,
        )

        # ----------------------------------------------------
        # VERIFY EVENT
        # ----------------------------------------------------

        assert result["event_preserved"] is True

        assert result["event_type"]

        assert math.isclose(
            result["event_timestamp_seconds"],
            TARGET_EVENT_SECONDS,
            abs_tol=0.1,
        )

        assert result["event_detection_source"] == (
            "recorded_reference_ecg"
        )

        assert (
            result["event_detected_from_corrupted_ecg"]
            is False
        )

        # ----------------------------------------------------
        # VERIFY WEARABLE CONTEXT
        # ----------------------------------------------------

        assert result["scenario"] == scenario

        assert result["sampling_rate"] == 250

        assert result["motion_is_simulated"] is True

        assert math.isfinite(
            result["motion_rms"]
        )

        # ----------------------------------------------------
        # VERIFY REVIEW LOGIC
        # ----------------------------------------------------

        assert result["review_status"] in {
            "REVIEW_EVENT",
            "REVIEW_SIGNAL_QUALITY",
        }

        assert result["requires_review"] is True

        assert result["clinical_validated"] is False

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print()
        print("-" * 60)

        print(
            "Scenario:",
            result["scenario"],
        )

        print(
            "Event type:",
            result["event_type"],
        )

        print(
            "Event timestamp:",
            result["event_timestamp_seconds"],
        )

        print(
            "RR interval:",
            result["rr_interval_seconds"],
        )

        print(
            "Baseline RR:",
            result["baseline_rr_seconds"],
        )

        print(
            "Motion RMS:",
            round(
                result["motion_rms"],
                4,
            ),
        )

        print(
            "Artefact prediction:",
            result["predicted_artifact_class"],
        )

        print(
            "Model probability:",
            result["artifact_model_probability"],
        )

        print(
            "Quality flags:",
            result["quality_flags"],
        )

        print(
            "Review status:",
            result["review_status"],
        )

        print(
            "Event preserved:",
            result["event_preserved"],
        )

        results.append(result)

    # --------------------------------------------------------
    # CROSS-SCENARIO VERIFICATION
    # --------------------------------------------------------

    timestamps = {
        result["event_timestamp_seconds"]
        for result in results
    }

    assert len(timestamps) == 1, (
        "Reference event timestamp changed "
        "between scenarios."
    )

    assert len(results) == len(SCENARIOS)

    print()
    print("=" * 60)

    print(
        "PASS: Event candidates preserved "
        "across virtual wearable scenarios."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
