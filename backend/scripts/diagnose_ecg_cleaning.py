
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window

DATA_FILE = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    with np.load(DATA_FILE, allow_pickle=False) as data:
        reference = data["ecg_reference"]
        corrupted = data["ecg_corrupted"]
        labels = data["labels"]
        class_names = [
            str(value) for value in data["class_names"]
        ]
        fs = int(data["fs"])

    indices = [
        i
        for i, label in enumerate(labels)
        if (
            class_names[int(label)]
            if isinstance(label, (int, np.integer))
            else str(label)
        ) == "baseline_wander"
    ]

    if not indices:
        raise RuntimeError("No baseline windows found.")

    index = indices[0]

    original = np.asarray(
        reference[index], dtype=np.float64
    )

    noisy = np.asarray(
        corrupted[index], dtype=np.float64
    )

    result = clean_ecg_window(
        noisy,
        fs,
        "baseline_wander",
        "REVIEW",
    )

    assert result.cleaned_ecg is not None

    filtered = result.cleaned_ecg

    # Diagnostic only: align the filtered signal's
    # median to the original reference median.
    aligned = (
        filtered
        + np.median(original)
        - np.median(filtered)
    )

    print("TRACE — BASELINE OFFSET DIAGNOSIS")
    print("--------------------------------")
    print(f"Window index: {index}")
    print(f"Sampling rate: {fs} Hz")

    print()
    print("Signal means:")
    print(f"Original: {np.mean(original):.4f}")
    print(f"Corrupted: {np.mean(noisy):.4f}")
    print(f"Filtered: {np.mean(filtered):.4f}")
    print(f"Aligned: {np.mean(aligned):.4f}")

    print()
    print("Signal ranges:")
    print(
        f"Original: "
        f"{np.min(original):.2f} to "
        f"{np.max(original):.2f}"
    )
    print(
        f"Filtered: "
        f"{np.min(filtered):.2f} to "
        f"{np.max(filtered):.2f}"
    )

    print()
    print("RMSE against recorded reference:")
    print(f"Raw: {rmse(noisy, original):.4f}")
    print(f"Filtered: {rmse(filtered, original):.4f}")
    print(f"Median-aligned: {rmse(aligned, original):.4f}")

    print()
    print("DIAGNOSIS COMPLETED")


if __name__ == "__main__":
    main()
