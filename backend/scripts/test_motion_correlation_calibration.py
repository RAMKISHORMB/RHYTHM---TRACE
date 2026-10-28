
"""
TRACE Stage 11.10
Motion-Correlation Calibration Verification.
"""

import csv
import math
from pathlib import Path

from app.motion_correlation_calibration import (
    calibrate_and_evaluate,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data"
    / "benchmarks"
    / "trace_motion_calibration.csv"
)


def main():
    print("=" * 70)
    print("TRACE STAGE 11.10 — MOTION CALIBRATION")
    print("=" * 70)

    results = []

    for informative in (True, False):
        for strength in (0.5, 1.0, 2.0):
            result = calibrate_and_evaluate(
                strength=strength,
                seed=42,
                informative=informative,
            )

            assert math.isfinite(
                result["reduction_percent"]
            )

            assert math.isfinite(
                result["validation_improvement"]
            )

            assert result["clinical_validated"] is False

            results.append(result)

            print()
            print("-" * 70)
            print(
                "Reference:",
                result["reference_type"],
            )
            print(
                "Strength:",
                result["strength"],
            )
            print(
                "Decision:",
                result["decision"],
            )
            print(
                "Validation improvement:",
                round(
                    result["validation_improvement"],
                    6,
                ),
            )
            print(
                "Correction fraction:",
                round(
                    result["correction_fraction"],
                    6,
                ),
            )
            print(
                "Raw error RMS:",
                round(
                    result["raw_error_rms"],
                    6,
                ),
            )
            print(
                "Output error RMS:",
                round(
                    result["output_error_rms"],
                    6,
                ),
            )
            print(
                "Artefact reduction (%):",
                round(
                    result["reduction_percent"],
                    3,
                ),
            )
            print(
                "Event agreement F1:",
                result["output_agreement_f1"],
            )
            print(
                "Missed:",
                result["missed_events"],
            )
            print(
                "Extra:",
                result["extra_events"],
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
    print("=" * 70)
    print("Experiments:", len(results))
    print("CSV:", OUTPUT)
    print("PASS: Motion calibration experiments executed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
