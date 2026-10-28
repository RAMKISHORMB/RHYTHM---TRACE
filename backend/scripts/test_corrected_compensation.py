
"""
TRACE Stage 11.7.2 Verification

Runs six corrected compensation experiments
and exports the results to CSV.
"""

import csv
import math
from pathlib import Path

from app.corrected_compensation_evaluation import (
    SCENARIOS,
    STRENGTHS,
    evaluate_corrected_compensation,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data"
    / "benchmarks"
    / "trace_corrected_compensation.csv"
)


def main():
    print("=" * 72)
    print("TRACE — STAGE 11.7.2")
    print("CORRECTED ECG COMPENSATION EVALUATION")
    print("=" * 72)

    results = []

    for scenario in SCENARIOS:
        for strength in STRENGTHS:

            result = evaluate_corrected_compensation(
                scenario=scenario,
                strength=strength,
                seed=42,
            )

            assert result["raw_artifact_rms"] > 0

            for key in (
                "raw_artifact_rms",
                "residual_artifact_rms",
                "artifact_reduction_percent",
                "clean_distortion_rms",
                "relative_clean_distortion_percent",
                "raw_agreement_f1",
                "filtered_agreement_f1",
            ):
                assert math.isfinite(
                    result[key]
                ), key

            assert (
                result["clinical_validated"]
                is False
            )

            results.append(result)

            print()
            print("-" * 72)

            print("Scenario:", scenario)
            print("Strength:", strength)

            print(
                "Raw artefact RMS:",
                result["raw_artifact_rms"],
            )

            print(
                "Residual artefact RMS:",
                result["residual_artifact_rms"],
            )

            print(
                "Artefact reduction (%):",
                result["artifact_reduction_percent"],
            )

            print(
                "Clean waveform distortion RMS:",
                result["clean_distortion_rms"],
            )

            print(
                "Relative clean distortion (%):",
                result[
                    "relative_clean_distortion_percent"
                ],
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
        "PASS: Corrected compensation "
        "evaluation executed."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
