
"""
TRACE — Stage 11.11
Motion Compensation Robustness Benchmark

Tests:
- Multiple random seeds
- Informative and unrelated motion references
- Different artefact strengths
- Clean ECG negative controls
- Event preservation
- Correction acceptance behavior

Uses the existing Stage 11.10 implementation.

Research prototype only.
"""

import csv
import math
from pathlib import Path

import numpy as np

from app.motion_correlation_calibration import (
    calibrate_and_evaluate,
    generate_case,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
    TAPS,
    RIDGE,
    MAX_CORRECTION_FRACTION,
    MIN_VALIDATION_IMPROVEMENT,
    lagged_matrix,
    rms,
)

from app.corrupted_event_evaluation import FS


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data"
    / "benchmarks"
    / "trace_motion_robustness.csv"
)

SEEDS = (11, 23, 42, 73, 101)
STRENGTHS = (0.5, 1.0, 2.0)


def evaluate_clean_control(seed):
    """
    Test whether the compensation gate attempts
    to modify a clean ECG when supplied with a
    nonzero, unrelated simulated motion signal.

    The clean ECG is used as the observed input,
    but no clean reference is supplied to the
    coefficient-fitting procedure.
    """

    reference, _, motion = generate_case(
        strength=1.0,
        seed=seed,
        informative=False,
    )

    observed = reference.copy()

    train_start = int(TRAIN_START * FS)
    train_end = int(TRAIN_END * FS)
    test_start = int(TEST_START * FS)
    test_end = int(TEST_END * FS)

    baseline = float(
        np.median(observed[train_start:train_end])
    )

    motion_mean = float(
        np.mean(motion[train_start:train_end])
    )

    centered_ecg = observed - baseline
    centered_motion = motion - motion_mean

    features = lagged_matrix(
        centered_motion,
        taps=TAPS,
    )

    split = train_start + int(
        0.75 * (train_end - train_start)
    )

    X_fit = features[train_start:split]
    y_fit = centered_ecg[train_start:split]

    X_validation = features[split:train_end]
    y_validation = centered_ecg[split:train_end]

    coefficients = np.linalg.solve(
        X_fit.T @ X_fit
        + RIDGE * np.eye(TAPS),
        X_fit.T @ y_fit,
    )

    validation_prediction = (
        X_validation @ coefficients
    )

    validation_improvement = (
        1.0
        - np.var(
            y_validation - validation_prediction
        )
        / max(
            np.var(y_validation),
            1e-12,
        )
    )

    predicted_artifact = features @ coefficients

    correction_fraction = (
        rms(
            predicted_artifact[
                test_start:test_end
            ]
        )
        / max(
            np.std(
                observed[test_start:test_end]
            ),
            1e-12,
        )
    )

    accepted = bool(
        validation_improvement
        >= MIN_VALIDATION_IMPROVEMENT
        and correction_fraction
        <= MAX_CORRECTION_FRACTION
    )

    output = observed.copy()

    if accepted:
        output[test_start:test_end] -= (
            predicted_artifact[
                test_start:test_end
            ]
        )

    clean_distortion = rms(
        output[test_start:test_end]
        - reference[test_start:test_end]
    )

    return {
        "test_type": "CLEAN_CONTROL",
        "reference_type": "UNRELATED",
        "strength": 0.0,
        "seed": seed,
        "accepted": accepted,
        "decision": (
            "ACCEPT_CORRECTION"
            if accepted
            else "PRESERVE_ORIGINAL"
        ),
        "validation_improvement": float(
            validation_improvement
        ),
        "correction_fraction": float(
            correction_fraction
        ),
        "raw_error_rms": 0.0,
        "output_error_rms": float(
            clean_distortion
        ),
        "reduction_percent": 0.0,
        "raw_agreement_f1": 1.0,
        "output_agreement_f1": None,
        "missed_events": None,
        "extra_events": None,
        "clinical_validated": False,
    }


def main():
    print("=" * 72)
    print("TRACE STAGE 11.11 — ROBUSTNESS BENCHMARK")
    print("=" * 72)

    results = []

    for seed in SEEDS:
        for strength in STRENGTHS:
            for informative in (True, False):

                result = calibrate_and_evaluate(
                    strength=strength,
                    seed=seed,
                    informative=informative,
                )

                result["test_type"] = (
                    "CORRUPTED_ECG"
                )

                results.append(result)

                print(
                    f"seed={seed} "
                    f"strength={strength} "
                    f"reference={result['reference_type']} "
                    f"decision={result['decision']} "
                    f"reduction={result['reduction_percent']:.2f}% "
                    f"F1={result['output_agreement_f1']:.3f}"
                )

    print()
    print("CLEAN ECG NEGATIVE CONTROLS")
    print("-" * 72)

    for seed in SEEDS:
        result = evaluate_clean_control(seed)

        results.append(result)

        print(
            f"seed={seed} "
            f"decision={result['decision']} "
            f"distortion_RMS={result['output_error_rms']:.6f}"
        )

    assert len(results) == 35

    for result in results:
        assert result["clinical_validated"] is False
        assert math.isfinite(
            result["output_error_rms"]
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "test_type",
        "reference_type",
        "strength",
        "seed",
        "accepted",
        "decision",
        "validation_improvement",
        "correction_fraction",
        "raw_error_rms",
        "output_error_rms",
        "reduction_percent",
        "raw_agreement_f1",
        "output_agreement_f1",
        "missed_events",
        "extra_events",
        "clinical_validated",
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    informative = [
        row
        for row in results
        if row["test_type"] == "CORRUPTED_ECG"
        and row["reference_type"] == "INFORMATIVE"
    ]

    unrelated = [
        row
        for row in results
        if row["test_type"] == "CORRUPTED_ECG"
        and row["reference_type"] == "UNRELATED"
    ]

    clean = [
        row
        for row in results
        if row["test_type"] == "CLEAN_CONTROL"
    ]

    accepted_informative = [
        row for row in informative
        if row["accepted"]
    ]

    accepted_unrelated = [
        row for row in unrelated
        if row["accepted"]
    ]

    accepted_clean = [
        row for row in clean
        if row["accepted"]
    ]

    print()
    print("=" * 72)
    print("ROBUSTNESS SUMMARY")
    print("=" * 72)

    print("Total experiments:", len(results))

    print(
        "Informative corrections accepted:",
        len(accepted_informative),
        "/",
        len(informative),
    )

    print(
        "Unrelated corrections accepted:",
        len(accepted_unrelated),
        "/",
        len(unrelated),
    )

    print(
        "Clean ECG corrections accepted:",
        len(accepted_clean),
        "/",
        len(clean),
    )

    if accepted_informative:
        mean_reduction = np.mean(
            [
                row["reduction_percent"]
                for row in accepted_informative
            ]
        )

        print(
            "Mean reduction in accepted "
            "informative cases:",
            round(float(mean_reduction), 3),
            "%",
        )

    print(
        "Worst clean ECG distortion RMS:",
        round(
            max(
                row["output_error_rms"]
                for row in clean
            ),
            6,
        ),
    )

    print(
        "Total missed event candidates:",
        sum(
            int(row["missed_events"])
            for row in results
            if row["missed_events"] is not None
        ),
    )

    print(
        "Total extra event candidates:",
        sum(
            int(row["extra_events"])
            for row in results
            if row["extra_events"] is not None
        ),
    )

    print("CSV saved:", OUTPUT)
    print()
    print(
        "PASS: All 35 robustness experiments executed."
    )


if __name__ == "__main__":
    main()
