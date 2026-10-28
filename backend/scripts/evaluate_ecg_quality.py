
"""
TRACE — ECG Quality Heuristic Evaluation

Evaluates provisional quality flags across the
complete synthetic artefact dataset.

This is descriptive testing, not clinical validation.
"""

from pathlib import Path
import sys
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_quality import analyze_ecg_quality


def main():
    path = ROOT / "data" / "processed" / "trace_artifacts.npz"

    with np.load(path, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        labels = data["labels"]
        names = data["class_names"]
        fs = int(data["fs"])

    print("\nTRACE — ECG QUALITY EVALUATION")
    print("=" * 78)

    print(
        f"{'CLASS':<20}"
        f"{'COUNT':>8}"
        f"{'ANY FLAG %':>14}"
        f"{'DRIFT %':>12}"
        f"{'STEP %':>12}"
        f"{'FLAT %':>12}"
    )
    print("-" * 78)

    results = {}

    for class_id, name in enumerate(names):
        indices = np.where(labels == class_id)[0]
        counts = Counter()

        for index in indices:
            result = analyze_ecg_quality(
                ecg[index],
                fs=fs,
            )

            flags = set(result.quality_flags)

            if flags:
                counts["any"] += 1

            if "POSSIBLE_BASELINE_DRIFT" in flags:
                counts["drift"] += 1

            if "POSSIBLE_ABRUPT_CHANGE" in flags:
                counts["step"] += 1

            if "POSSIBLE_LOCAL_DROPOUT" in flags:
                counts["flat"] += 1

        total = len(indices)
        results[str(name)] = {
            key: counts[key] / total
            for key in ("any", "drift", "step", "flat")
        }

        print(
            f"{str(name):<20}"
            f"{total:>8}"
            f"{100 * counts['any'] / total:>13.1f}%"
            f"{100 * counts['drift'] / total:>11.1f}%"
            f"{100 * counts['step'] / total:>11.1f}%"
            f"{100 * counts['flat'] / total:>11.1f}%"
        )

    print("\nIMPORTANT:")
    print(
        "Clean-class warning rate measures false "
        "warnings within this synthetic dataset."
    )
    print(
        "Artefact-class flag rates are detection "
        "rates for these synthetic artefact types."
    )
    print(
        "Results are not estimates of clinical "
        "sensitivity or specificity."
    )


if __name__ == "__main__":
    main()
