
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.trust_engine import evaluate_trust


def main():
    rng = np.random.default_rng(42)

    # Synthetic non-flatline test signal.
    signal = rng.normal(
        loc=1000,
        scale=30,
        size=1000,
    )

    cases = [
        ("clean", 0.98, "HIGH"),
        ("baseline_wander", 0.94, "REVIEW"),
        ("walking", 0.91, "REVIEW"),
        ("electrode_pop", 0.97, "UNRELIABLE"),
        ("lead_dropout", 0.99, "UNRELIABLE"),
        ("clean", 0.40, "REVIEW"),
    ]

    for name, probability, expected in cases:
        result = evaluate_trust(
            ecg_window=signal,
            predicted_class=name,
            model_probability=probability,
        )

        print(
            f"{name:18} "
            f"prob={probability:.2f} "
            f"trust={result.level}"
        )

        assert result.level == expected, (
            f"Expected {expected}, got {result.level}"
        )

    # Test dropout override on a flat signal.
    flat_signal = np.full(1000, 1000.0)

    result = evaluate_trust(
        ecg_window=flat_signal,
        predicted_class="clean",
        model_probability=0.99,
    )

    assert result.level == "UNRELIABLE"

    print(
        "flatline override  "
        f"trust={result.level}"
    )

    print("\nTRACE TRUST ENGINE TEST PASSED")


if __name__ == "__main__":
    main()
