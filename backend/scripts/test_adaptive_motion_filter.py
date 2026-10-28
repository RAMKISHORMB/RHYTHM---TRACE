
"""
TRACE — Stage 11.8
Adaptive Motion Filter Verification

Evaluates:
- Synthetic motion-correlated artefact reduction
- Transient event preservation
- Clean ECG distortion
- Numerical stability

No clinical performance claims.
"""

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)

from app.adaptive_motion_filter import (
    nlms_motion_cancel,
)


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def run_case(strength, seed=42):
    rng = np.random.default_rng(seed)

    reference = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    n = len(reference)
    time = np.arange(n, dtype=np.float64) / FS

    # Synthetic motion reference.
    motion = (
        np.sin(2 * np.pi * 1.8 * time)
        + 0.4 * np.sin(2 * np.pi * 0.65 * time)
        + 0.15 * rng.normal(size=n)
    )

    # Synthetic motion-correlated interference.
    delayed_motion = np.concatenate(
        (
            np.zeros(3),
            motion[:-3],
        )
    )

    artifact = strength * (
        0.15 * motion
        + 0.06 * delayed_motion
    )

    observed = reference + artifact

    output = nlms_motion_cancel(
        observed_ecg=observed,
        motion_reference=motion,
        fs=FS,
    )

    compensated = output["compensated_ecg"]

    # Apply the same adaptive procedure to clean
    # ECG to measure unintended signal changes.
    clean_output = nlms_motion_cancel(
        observed_ecg=reference,
        motion_reference=motion,
        fs=FS,
    )

    raw_error = rms(observed - reference)
    compensated_error = rms(compensated - reference)
    clean_distortion = rms(
        clean_output["compensated_ecg"] - reference
    )

    reduction_percent = (
        100.0 * (raw_error - compensated_error) / raw_error
        if raw_error > 0
        else 0.0
    )

    reference_events = detect_events(reference)
    raw_events = detect_events(observed)
    compensated_events = detect_events(compensated)

    raw_match = match_events(
        reference_events,
        raw_events,
    )

    compensated_match = match_events(
        reference_events,
        compensated_events,
    )

    assert np.isfinite(compensated).all()
    assert np.isfinite(clean_output["compensated_ecg"]).all()

    return {
        "strength": strength,
        "seed": seed,
        "raw_error_rms": raw_error,
        "compensated_error_rms": compensated_error,
        "reduction_percent": reduction_percent,
        "clean_distortion_rms": clean_distortion,
        "raw_agreement_f1": raw_match["agreement_f1"],
        "compensated_agreement_f1": compensated_match[
            "agreement_f1"
        ],
        "compensated_missed": compensated_match[
            "unmatched_reference_events"
        ],
        "compensated_extra": compensated_match[
            "unmatched_observed_events"
        ],
    }


def main():
    print("=" * 68)
    print("TRACE STAGE 11.8 — ADAPTIVE MOTION FILTER")
    print("=" * 68)

    results = []

    for strength in (0.5, 1.0, 2.0):
        result = run_case(
            strength=strength,
            seed=42,
        )

        results.append(result)

        print()
        print("-" * 68)
        print("Strength:", result["strength"])

        print(
            "Raw error RMS:",
            round(result["raw_error_rms"], 6),
        )

        print(
            "Compensated error RMS:",
            round(result["compensated_error_rms"], 6),
        )

        print(
            "Artefact reduction (%):",
            round(result["reduction_percent"], 3),
        )

        print(
            "Clean ECG distortion RMS:",
            round(result["clean_distortion_rms"], 6),
        )

        print(
            "Raw agreement F1:",
            result["raw_agreement_f1"],
        )

        print(
            "Compensated agreement F1:",
            result["compensated_agreement_f1"],
        )

        print(
            "Compensated missed events:",
            result["compensated_missed"],
        )

        print(
            "Compensated extra events:",
            result["compensated_extra"],
        )

    assert len(results) == 3

    print()
    print("=" * 68)
    print("PASS: All 3 adaptive-filter experiments executed.")
    print("=" * 68)


if __name__ == "__main__":
    main()
