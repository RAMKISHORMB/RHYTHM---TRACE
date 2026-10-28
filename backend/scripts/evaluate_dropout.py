
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dropout_detection import detect_local_dropout


def main():
    path = ROOT / "data" / "processed" / "trace_artifacts.npz"

    with np.load(path, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        labels = data["labels"]
        names = data["class_names"]
        fs = int(data["fs"])

    print("\nTRACE — LOCAL DROPOUT EVALUATION")
    print("=" * 74)

    print(
        f"{'CLASS':<20}"
        f"{'COUNT':>8}"
        f"{'FLAGGED %':>14}"
        f"{'MEDIAN RATIO':>16}"
        f"{'MEDIAN LOW MS':>16}"
    )
    print("-" * 74)

    for class_id, name in enumerate(names):
        indices = np.where(labels == class_id)[0]

        flagged = 0
        ratios = []
        durations = []

        for index in indices:
            result = detect_local_dropout(
                ecg[index],
                fs=fs,
            )

            flagged += int(result.possible_dropout)
            ratios.append(
                result.minimum_amplitude_ratio
            )
            durations.append(
                result.longest_low_amplitude_ms
            )

        print(
            f"{str(name):<20}"
            f"{len(indices):>8}"
            f"{100 * flagged / len(indices):>13.1f}%"
            f"{np.median(ratios):>16.4f}"
            f"{np.median(durations):>16.1f}"
        )

    print("\nEvaluation complete.")
    print(
        "Synthetic data only; no clinical "
        "sensitivity or specificity established."
    )


if __name__ == "__main__":
    main()
