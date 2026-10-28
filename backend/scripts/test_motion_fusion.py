
"""
TRACE Stage 6.3.1
Controlled ECG–motion fusion counterexample tests.

Uses synthetic test signals only.
Does not test cardiac event detection.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.motion_fusion import analyze_ecg_motion


def main():
    fs = 250
    t = np.arange(1000) / fs

    # Simple illustrative waveform, not clinical ECG.
    reference = 100 * np.sin(2 * np.pi * 1.2 * t)

    # Scenario 1: High motion, no ECG corruption.
    high_motion = 0.5 * np.sin(2 * np.pi * 2 * t)

    result = analyze_ecg_motion(
        reference,
        reference.copy(),
        high_motion,
    )

    assert result.motion_level == "HIGH"
    assert np.isclose(result.ecg_difference_rms, 0)
    assert result.temporal_correlation is None
    assert result.evidence_category == (
        "MOTION_PRESENT_UNCONFIRMED"
    )
    print("PASS 1: High motion does not imply ECG corruption")

    # Scenario 2: Low motion, substantial ECG corruption.
    low_motion = 0.01 * np.sin(2 * np.pi * 2 * t)
    corrupted = reference + 40 * np.sin(
        2 * np.pi * 0.3 * t
    )

    result = analyze_ecg_motion(
        reference,
        corrupted,
        low_motion,
    )

    assert result.motion_level == "LOW"
    assert result.ecg_difference_rms > 10
    assert result.evidence_category == (
        "LOW_MOTION_EVIDENCE"
    )
    print("PASS 2: ECG corruption can occur with low motion")

    # Scenario 3: No motion and no corruption.
    stationary = np.zeros_like(t)

    result = analyze_ecg_motion(
        reference,
        reference.copy(),
        stationary,
    )

    assert result.temporal_correlation is None
    assert result.ecg_difference_rms == 0
    print("PASS 3: Constant signals handled safely")

    # Scenario 4: Reject mismatched sample counts.
    try:
        analyze_ecg_motion(
            reference,
            reference,
            stationary[:-1],
        )
    except ValueError:
        print("PASS 4: Misaligned signals rejected")
    else:
        raise AssertionError(
            "Misaligned motion was not rejected"
        )

    # Scenario 5: Reject non-finite data.
    invalid_motion = stationary.copy()
    invalid_motion[100] = np.nan

    try:
        analyze_ecg_motion(
            reference,
            reference,
            invalid_motion,
        )
    except ValueError:
        print("PASS 5: Invalid sensor data rejected")
    else:
        raise AssertionError(
            "Non-finite data was not rejected"
        )

    print()
    print("ALL 5 MOTION FUSION TESTS PASSED")


if __name__ == "__main__":
    main()
