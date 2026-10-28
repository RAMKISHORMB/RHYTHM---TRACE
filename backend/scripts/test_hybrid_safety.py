
"""
TRACE — Hybrid Fusion Safety Tests
MEDAITHON'26 | PS-001

Tests engineering behavior only.
Does not establish clinical safety or performance.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hybrid_fusion import analyze_hybrid_fusion


def main():
    fs = 250

    with np.load(
        ROOT / "data" / "processed" / "trace_artifacts.npz",
        allow_pickle=False,
    ) as data:
        ecg = data["ecg_corrupted"]
        imu = data["imu_simulated"]
        labels = data["labels"]
        names = [str(x) for x in data["class_names"]]

    def example(class_name):
        class_id = names.index(class_name)
        index = int(np.where(labels == class_id)[0][0])
        return ecg[index], imu[index]

    # TEST 1: Motion alone must not imply dropout.
    walking_ecg, walking_imu = example("walking")

    result = analyze_hybrid_fusion(
        walking_ecg,
        walking_imu,
        fs=fs,
    )

    assert result.motion_level == "HIGH"
    assert result.possible_dropout is False
    print("PASS 1: Motion does not automatically imply dropout")

    # TEST 2: Preserve disagreement on a clean example.
    clean_ecg, clean_imu = example("clean")

    result = analyze_hybrid_fusion(
        clean_ecg,
        clean_imu,
        fs=fs,
        artifact_prediction="clean",
        artifact_confidence=1.0,
    )

    assert result.evidence_conflict is True
    assert result.decision == "REVIEW_CONFLICT"
    print("PASS 2: Classifier/quality disagreement preserved")

    # TEST 3: Strong dropout evidence cannot be
    # cleared by a clean classifier prediction.
    dropout_ecg, dropout_imu = example("lead_dropout")

    result = analyze_hybrid_fusion(
        dropout_ecg,
        dropout_imu,
        fs=fs,
        artifact_prediction="clean",
        artifact_confidence=1.0,
    )

    assert result.possible_dropout is True
    assert result.evidence_conflict is True
    assert result.decision == "REVIEW_CONFLICT"
    print("PASS 3: Clean prediction cannot override dropout")

    # TEST 4: Reject sensor misalignment.
    try:
        analyze_hybrid_fusion(
            clean_ecg,
            clean_imu[:-1],
            fs=fs,
        )
    except ValueError:
        print("PASS 4: Misaligned sensors rejected")
    else:
        raise AssertionError("Misalignment was accepted")

    # TEST 5: Reject non-finite ECG samples.
    invalid_ecg = clean_ecg.copy()
    invalid_ecg[10] = np.nan

    try:
        analyze_hybrid_fusion(
            invalid_ecg,
            clean_imu,
            fs=fs,
        )
    except ValueError:
        print("PASS 5: Invalid ECG samples rejected")
    else:
        raise AssertionError("Invalid ECG was accepted")

    # TEST 6: Preserve the cardiac-event safety policy.
    result = analyze_hybrid_fusion(
        walking_ecg,
        walking_imu,
        fs=fs,
        artifact_prediction="walking",
    )

    assert "Never automatically dismiss" in (
        result.cardiac_event_policy
    )
    assert result.experimental is True
    assert result.clinical_validated is False

    print("PASS 6: Cardiac-event policy retained")

    print()
    print("ALL 6 HYBRID SAFETY TESTS PASSED")
    print(
        "Note: These are software assertions, "
        "not clinical safety validation."
    )


if __name__ == "__main__":
    main()
