
"""
TRACE — Dropout Duration Threshold Analysis

Compares duration thresholds on the synthetic
artefact dataset without modifying the detector.

Exploratory analysis only. Do not interpret
these results as independent validation.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dropout_detection import detect_local_dropout


THRESHOLDS_MS = [120, 160, 200, 250, 300, 400, 500]


def main():
    path = ROOT / "data" / "processed" / "trace_artifacts.npz"

    with np.load(path, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        labels = data["labels"]
        names = data["class_names"]
        fs = int(data["fs"])

    durations = np.zeros(len(ecg), dtype=np.float64)

    print("\nAnalyzing ECG windows...")

    for index, window in enumerate(ecg):
        result = detect_local_dropout(window, fs=fs)
        durations[index] = result.longest_low_amplitude_ms

    print("\nTRACE — DROPOUT DURATION THRESHOLD ANALYSIS")
    print("=" * 88)

    print(
        f"{'THRESHOLD':<13}"
        + "".join(
            f"{str(name)[:14]:>15}"
            for name in names
        )
    )
    print("-" * 88)

    for threshold in THRESHOLDS_MS:
        row = f"{threshold:<13}"

        for class_id in range(len(names)):
            mask = labels == class_id
            flagged = np.mean(
                durations[mask] >= threshold
            ) * 100

            row += f"{flagged:>14.1f}%"

        print(row)

    print("\nDURATION DISTRIBUTION BY CLASS")
    print("-" * 65)
    print(
        f"{'CLASS':<20}"
        f"{'P50':>12}"
        f"{'P90':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
    )

    for class_id, name in enumerate(names):
        values = durations[labels == class_id]

        print(
            f"{str(name):<20}"
            f"{np.percentile(values, 50):>12.1f}"
            f"{np.percentile(values, 90):>12.1f}"
            f"{np.percentile(values, 95):>12.1f}"
            f"{np.percentile(values, 99):>12.1f}"
        )

    print("\nResearch limitations:")
    print("- Synthetic ECG artefact windows only.")
    print("- Windows may overlap or share source recordings.")
    print("- Threshold selection here is exploratory.")
    print("- Independent-record validation is still required.")


if __name__ == "__main__":
    main()
