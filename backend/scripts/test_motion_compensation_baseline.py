
"""
TRACE Stage 11.7 — Compensation Baseline Test

Six experiments:
2 artefact types x 3 strength levels.

The test passes if evaluation executes and
produces valid measurements.

It does not require filtering to improve results.
"""

import csv
import math
from pathlib import Path

from app.motion_compensation_baseline import (
    SCENARIOS,
    STRENGTHS,
    evaluate_compensation,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data"
    / "benchmarks"
    / "trace_compensation_baseline.csv"
)


def main():
    print("=" * 72)
    print("TRACE — STAGE 11.7")
    print("MOTION COMPENSATION BASELINE")
    print("=" * 72)

    results = []

    for scenario in SCENARIOS:
        for strength in STRENGTHS:

            result = evaluate_compensation(
                scenario=scenario,
                strength=strength,
                seed=42,
            )

            assert result["raw_error_rms"] > 0

            for key in (
                "raw_error_rms",
                "filtered_error_rms",
                "improvement_percent",
                "clean_filter_distortion_rms",
                "residual_artifact_rms",
                "raw_agreement_f1",
                "filtered_agreement_f1",
            ):
                assert math.isfinite(result[key]), key

            assert result["clinical_validated"] is False

            results.append(result)

            print()
            print("Scenario:", scenario)
            print("Strength:", strength)

            print(
                "Raw error RMS:",
                result["raw_error_rms"],
            )

            print(
                "Filtered error RMS:",
                result["filtered_error_rms"],
            )

            print(
                "Improvement (%):",
                result["improvement_percent"],
            )

            print(
                "Clean ECG distortion RMS:",
                result["clean_filter_distortion_rms"],
            )

            print(
                "Raw event agreement F1:",
                result["raw_agreement_f1"],
            )

            print(
                "Filtered event agreement F1:",
                result["filtered_agreement_f1"],
            )

            print(
                "Filtered missed events:",
                result["filtered_missed_events"],
            )

            print(
                "Filtered extra events:",
                result["filtered_extra_events"],
            )

    assert len(results) == 6

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 72)
    print("Total experiments:", len(results))
    print("CSV saved:", OUTPUT)
    print(
        "PASS: Motion compensation baseline "
        "evaluation executed."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
