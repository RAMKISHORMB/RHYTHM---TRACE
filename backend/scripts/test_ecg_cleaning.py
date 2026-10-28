
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window


def main() -> None:
    fs = 250
    t = np.arange(1000) / fs

    # Synthetic test waveform, not a physiological ECG.
    reference = np.sin(2 * np.pi * 5 * t)

    baseline = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    corrupted = reference + baseline

    # TEST 1 — Preserve high-trust signal
    result = clean_ecg_window(
        reference,
        fs,
        "clean",
        "HIGH",
    )

    assert result.status == "PRESERVED"
    assert result.cleaned_ecg is not None
    assert np.array_equal(
        result.raw_ecg,
        result.cleaned_ecg,
    )

    print("TEST 1 PASSED: HIGH preserves signal")

    # TEST 2 — Baseline wander candidate filtering
    result = clean_ecg_window(
        corrupted,
        fs,
        "baseline_wander",
        "REVIEW",
    )

    assert result.status == "CANDIDATE_FILTERED"
    assert result.cleaned_ecg is not None
    assert result.cleaned_ecg.shape == corrupted.shape
    assert np.all(np.isfinite(result.cleaned_ecg))

    raw_rmse = np.sqrt(
        np.mean((corrupted - reference) ** 2)
    )

    cleaned_rmse = np.sqrt(
        np.mean((result.cleaned_ecg - reference) ** 2)
    )

    print(f"Raw RMSE:     {raw_rmse:.4f}")
    print(f"Cleaned RMSE: {cleaned_rmse:.4f}")

    assert cleaned_rmse < raw_rmse

    print("TEST 2 PASSED: Baseline candidate improves RMSE")

    # TEST 3 — Unreliable signal: abstain
    result = clean_ecg_window(
        corrupted,
        fs,
        "lead_dropout",
        "UNRELIABLE",
    )

    assert result.status == "ABSTAINED"
    assert result.cleaned_ecg is None

    print("TEST 3 PASSED: UNRELIABLE abstains")

    # TEST 4 — Other REVIEW artefacts: preserve
    result = clean_ecg_window(
        corrupted,
        fs,
        "walking",
        "REVIEW",
    )

    assert result.status == "PRESERVED"
    assert result.cleaned_ecg is not None
    assert np.array_equal(
        result.cleaned_ecg,
        corrupted,
    )

    print("TEST 4 PASSED: Unsupported filtering avoided")

    # TEST 5 — Invalid input
    try:
        clean_ecg_window(
            np.array([1.0, np.nan]),
            fs,
            "clean",
            "HIGH",
        )
    except ValueError:
        print("TEST 5 PASSED: Invalid input rejected")
    else:
        raise AssertionError("NaN input was accepted")

    print()
    print("ALL 5 ECG CLEANING TESTS PASSED")


if __name__ == "__main__":
    main()
