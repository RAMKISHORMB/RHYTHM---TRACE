
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.audit_engine import create_audit_entry


def main():
    high = create_audit_entry(
        0, "clean", "HIGH", "PRESERVED"
    )
    assert high["action"] == "PRESERVED"
    print("TEST 1 PASSED: HIGH preserves ECG")

    review = create_audit_entry(
        1,
        "baseline_wander",
        "REVIEW",
        "CANDIDATE_FILTERED",
    )
    assert review["experimental"] is True
    assert review["clinical_validated"] is False
    print("TEST 2 PASSED: Experimental cleaning audited")

    unreliable = create_audit_entry(
        2, "lead_dropout", "UNRELIABLE", "ABSTAINED"
    )
    assert unreliable["action"] == "ABSTAINED"
    print("TEST 3 PASSED: UNRELIABLE abstains")

    try:
        create_audit_entry(
            3, "lead_dropout", "UNRELIABLE", "PRESERVED"
        )
        raise AssertionError("Unsafe action was accepted.")
    except ValueError:
        print("TEST 4 PASSED: Unsafe action rejected")

    try:
        create_audit_entry(
            -1, "clean", "HIGH", "PRESERVED"
        )
        raise AssertionError("Invalid index was accepted.")
    except ValueError:
        print("TEST 5 PASSED: Invalid index rejected")

    print()
    print("AUDIT ENTRY EXAMPLE")
    for key, value in review.items():
        print(f"{key}: {value}")

    print()
    print("ALL 5 AUDIT ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
