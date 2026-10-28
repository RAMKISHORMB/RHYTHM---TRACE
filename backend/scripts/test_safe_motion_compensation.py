
"""
TRACE — Stage 11.9
Safe Motion Compensation Verification

Tests:
1. Clean ECG with weak motion reference
2. Motion-correlated synthetic interference
3. Uninformative motion reference

Reports signal error, clean-signal distortion,
acceptance decisions, and event agreement.
"""

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)

from app.safe_motion_compensation import (
    safe_motion_compensate,
    calculate_rms,
)


def make_motion(n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS

    return (
        np.sin(2 * np.pi * 1.8 * t)
        + 0.4 * np.sin(2 * np.pi * 0.65 * t)
        + 0.15 * rng.normal(size=n)
    )


def evaluate_case(name, reference, observed, motion):
    result = safe_motion_compensate(
        observed_ecg=observed,
        motion_reference=motion,
        fs=FS,
    )

    output = result["output_ecg"]

    raw_error = calculate_rms(
        observed - reference
    )

    output_error = calculate_rms(
        output - reference
    )

    reference_events = detect_events(reference)
    output_events = detect_events(output)

    comparison = match_events(
        reference_events,
        output_events,
    )

    assert len(output) == len(reference)
    assert np.isfinite(output).all()

    print()
    print("-" * 65)
    print("Scenario:", name)
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])
    print(
        "Motion RMS:",
        round(result["motion_rms"], 6),
    )
    print(
        "Explained variance:",
        round(result["explained_variance"], 6),
    )
    print(
        "Correction fraction:",
        round(result["correction_fraction"], 6),
    )
    print(
        "Raw error RMS:",
        round(raw_error, 6),
    )
    print(
        "Output error RMS:",
        round(output_error, 6),
    )
    print(
        "Event agreement F1:",
        comparison["agreement_f1"],
    )
    print(
        "Missed events:",
        comparison["unmatched_reference_events"],
    )
    print(
        "Extra events:",
        comparison["unmatched_observed_events"],
    )

    return result, raw_error, output_error


def main():
    print("=" * 65)
    print("TRACE STAGE 11.9 — SAFE MOTION COMPENSATION")
    print("=" * 65)

    reference = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    n = len(reference)
    motion = make_motion(n, seed=42)

    # Case 1: Clean ECG and negligible motion.
    clean_result, _, clean_error = evaluate_case(
        name="CLEAN_WEAK_MOTION",
        reference=reference,
        observed=reference.copy(),
        motion=np.zeros(n),
    )

    assert not clean_result["accepted"]
    assert clean_error == 0.0

    # Case 2: Motion-correlated synthetic artefact.
    delayed = np.concatenate(
        (np.zeros(3), motion[:-3])
    )

    artifact = (
        0.15 * motion
        + 0.06 * delayed
    )

    observed = reference + artifact

    correlated_result, _, _ = evaluate_case(
        name="CORRELATED_MOTION",
        reference=reference,
        observed=observed,
        motion=motion,
    )

    # Case 3: Corruption with unrelated motion.
    independent_motion = make_motion(
        n,
        seed=999,
    )

    uninformative_result, _, _ = evaluate_case(
        name="UNINFORMATIVE_REFERENCE",
        reference=reference,
        observed=observed,
        motion=independent_motion,
    )

    assert (
        correlated_result["clinical_validated"]
        is False
    )

    assert (
        uninformative_result["clinical_validated"]
        is False
    )

    print()
    print("=" * 65)
    print("PASS: All 3 safety-gated experiments executed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
