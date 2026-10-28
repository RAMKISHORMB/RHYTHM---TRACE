
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.beat_trust import evaluate_beat_trust


def main():
    fs = 250

    # Synthetic, varying waveform.
    t = np.arange(1000) / fs

    ecg = (
        1000.0
        + 60.0 * np.sin(2 * np.pi * 5 * t)
    )

    beats = [125, 375, 625, 875]
    symbols = ["N", "N", "N", "N"]

    # Case 1: Clean varying signal.
    results = evaluate_beat_trust(
        ecg,
        beats,
        symbols,
        fs,
        "HIGH",
    )

    assert len(results) == 4
    assert all(
        item.level == "HIGH"
        for item in results
    )

    print("Case 1: clean beat regions -> HIGH")

    # Case 2: A flat region around beat 2.
    corrupted = ecg.copy()
    corrupted[325:426] = 1000.0

    results = evaluate_beat_trust(
        corrupted,
        beats,
        symbols,
        fs,
        "HIGH",
    )

    assert results[1].level == "UNRELIABLE"
    assert results[0].level == "HIGH"
    assert results[2].level == "HIGH"
    assert results[3].level == "HIGH"

    print(
        "Case 2: local flatline -> "
        "affected beat UNRELIABLE"
    )

    # Case 3: Upstream window trust is REVIEW.
    results = evaluate_beat_trust(
        ecg,
        beats,
        symbols,
        fs,
        "REVIEW",
    )

    assert all(
        item.level == "REVIEW"
        for item in results
    )

    print(
        "Case 3: inherited REVIEW -> "
        "no automatic upgrade"
    )

    # Case 4: Upstream window is UNRELIABLE.
    results = evaluate_beat_trust(
        ecg,
        beats,
        symbols,
        fs,
        "UNRELIABLE",
    )

    assert all(
        item.level == "UNRELIABLE"
        for item in results
    )

    print(
        "Case 4: inherited UNRELIABLE -> "
        "no automatic upgrade"
    )

    # Case 5: Reject out-of-range annotations.
    try:
        evaluate_beat_trust(
            ecg,
            [2000],
            ["N"],
            fs,
            "HIGH",
        )
    except ValueError:
        print(
            "Case 5: invalid beat location rejected"
        )
    else:
        raise AssertionError(
            "Out-of-range beat should be rejected."
        )

    print("\nTRACE BEAT TRUST TEST PASSED")


if __name__ == "__main__":
    main()
