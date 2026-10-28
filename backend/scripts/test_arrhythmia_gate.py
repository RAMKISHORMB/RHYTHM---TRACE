
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.arrhythmia_gate import (
    verify_arrhythmia_prediction,
)


def expect_error(**kwargs):
    try:
        verify_arrhythmia_prediction(**kwargs)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError was not raised."
    )


def main():
    high = verify_arrhythmia_prediction(
        beat_sample=100,
        trust_level="HIGH",
        model_prediction="NORMAL",
        model_probability=0.92,
    )

    assert high.decision == "ACCEPT"
    assert high.released_prediction == "NORMAL"
    print("TEST 1 PASSED: HIGH accepts model output")

    review = verify_arrhythmia_prediction(
        beat_sample=200,
        trust_level="REVIEW",
        model_prediction="PVC",
        model_probability=0.81,
    )

    assert review.decision == "REVIEW"
    assert review.released_prediction is None
    print("TEST 2 PASSED: REVIEW withholds output")

    unreliable = verify_arrhythmia_prediction(
        beat_sample=300,
        trust_level="UNRELIABLE",
        model_prediction="PVC",
        model_probability=0.99,
    )

    assert unreliable.decision == "ABSTAIN"
    assert unreliable.released_prediction is None
    print("TEST 3 PASSED: UNRELIABLE abstains")

    missing = verify_arrhythmia_prediction(
        beat_sample=400,
        trust_level="HIGH",
        model_prediction=None,
        model_probability=None,
    )

    assert missing.decision == "ABSTAIN"
    print("TEST 4 PASSED: Missing prediction abstains")

    expect_error(
        beat_sample=500,
        trust_level="HIGH",
        model_prediction="NORMAL",
        model_probability=1.5,
    )
    print("TEST 5 PASSED: Invalid probability rejected")

    expect_error(
        beat_sample=600,
        trust_level="HIGH",
        model_prediction="NORMAL",
        model_probability=None,
    )
    print("TEST 6 PASSED: Incomplete prediction rejected")

    expect_error(
        beat_sample=-1,
        trust_level="HIGH",
        model_prediction="NORMAL",
        model_probability=0.9,
    )
    print("TEST 7 PASSED: Invalid beat sample rejected")

    print()
    print("ALL 7 ARRHYTHMIA GATE TESTS PASSED")


if __name__ == "__main__":
    main()
