
"""
TRACE — Stage 11.4 Verification

Compares transient-event detection on
reference ECG and simulated corrupted ECG.
"""

import math

from app.corrupted_event_evaluation import (
    SCENARIOS,
    evaluate_corrupted_scenario,
)


def main():

    print("=" * 60)
    print("TRACE — STAGE 11.4")
    print("REFERENCE VS CORRUPTED ECG")
    print("=" * 60)

    for scenario in SCENARIOS:

        result = evaluate_corrupted_scenario(
            scenario=scenario,
            target_event_seconds=185.532,
            seed=42,
        )

        comparison = result["comparison"]

        assert result["sampling_rate"] == 250

        assert result["duration_seconds"] == 300

        assert result["reference_event_count"] == 3

        assert math.isfinite(
            result["reference_corruption_rms"]
        )

        assert (
            comparison["matched_events"]
            + comparison[
                "unmatched_reference_events"
            ]
            == result["reference_event_count"]
        )

        assert (
            comparison["matched_events"]
            + comparison[
                "unmatched_observed_events"
            ]
            == result["observed_event_count"]
        )

        assert isinstance(
            result["target_event_detected"],
            bool,
        )

        assert result["clinical_validated"] is False

        print()
        print("-" * 60)

        print("Scenario:", scenario)

        print(
            "Reference events:",
            result["reference_event_count"],
        )

        print(
            "Observed events:",
            result["observed_event_count"],
        )

        print(
            "Corruption RMS:",
            round(
                result["reference_corruption_rms"],
                4,
            ),
        )

        print(
            "Matched events:",
            comparison["matched_events"],
        )

        print(
            "Unmatched reference:",
            comparison[
                "unmatched_reference_events"
            ],
        )

        print(
            "Unmatched observed:",
            comparison[
                "unmatched_observed_events"
            ],
        )

        print(
            "Agreement F1:",
            comparison["agreement_f1"],
        )

        print(
            "Target event detected:",
            result["target_event_detected"],
        )

        print("Matches:")

        for match in comparison["matches"]:
            print(
                "  ",
                match["event_type"],
                "reference=",
                match[
                    "reference_timestamp_seconds"
                ],
                "observed=",
                match[
                    "observed_timestamp_seconds"
                ],
                "error_ms=",
                match["timing_error_ms"],
            )

    print()
    print("=" * 60)

    print(
        "PASS: Reference and corrupted ECG "
        "comparison executed successfully."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
