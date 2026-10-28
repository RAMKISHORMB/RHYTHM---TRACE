
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window

DATA_FILE = ROOT / "data" / "processed" / "trace_artifacts.npz"

WORST_INDICES = [866, 2191, 1896, 1391, 1901]


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def linear_clean(ecg):
    ecg = np.asarray(ecg, dtype=np.float64)

    # Remove linear trend while preserving input mean.
    return (
        detrend(ecg, type="linear")
        + float(np.mean(ecg))
    )


def median_clean(ecg, fs):
    ecg = np.asarray(ecg, dtype=np.float64)

    # One-second odd-sized window at the sampling rate.
    size = fs if fs % 2 == 1 else fs + 1

    estimated_baseline = median_filter(
        ecg,
        size=size,
        mode="reflect",
    )

    # Preserve an input-derived baseline level.
    return (
        ecg
        - estimated_baseline
        + float(np.median(ecg))
    )


def main():
    with np.load(DATA_FILE, allow_pickle=False) as data:
        references = data["ecg_reference"]
        corrupted = data["ecg_corrupted"]
        labels = data["labels"]
        names = [str(x) for x in data["class_names"]]
        fs = int(data["fs"])

    indices = [
        i for i, label in enumerate(labels)
        if (
            names[int(label)]
            if isinstance(label, (int, np.integer))
            else str(label)
        ) == "baseline_wander"
    ]

    if not indices:
        raise RuntimeError("No baseline-wander windows found.")

    methods = [
        "raw",
        "butterworth",
        "linear",
        "median",
    ]

    errors = {name: [] for name in methods}
    per_window = {}

    print("TRACE — CLEANING METHOD COMPARISON")
    print("---------------------------------")
    print(f"Sampling rate: {fs} Hz")
    print(f"Baseline windows: {len(indices)}")

    for index in indices:
        reference = np.asarray(
            references[index], dtype=np.float64
        )
        noisy = np.asarray(
            corrupted[index], dtype=np.float64
        )

        butter_result = clean_ecg_window(
            noisy,
            fs,
            "baseline_wander",
            "REVIEW",
        )

        if butter_result.cleaned_ecg is None:
            raise RuntimeError(
                f"Butterworth returned no output: {index}"
            )

        outputs = {
            "raw": noisy,
            "butterworth": butter_result.cleaned_ecg,
            "linear": linear_clean(noisy),
            "median": median_clean(noisy, fs),
        }

        row = {}

        for name, output in outputs.items():
            if (
                output.shape != reference.shape
                or not np.all(np.isfinite(output))
            ):
                raise ValueError(
                    f"Invalid {name} output at {index}"
                )

            error = rmse(output, reference)
            errors[name].append(error)
            row[name] = error

        per_window[index] = row

    print("\nALL BASELINE WINDOWS")
    print(
        f"{'Method':14s} "
        f"{'Mean RMSE':>12s} "
        f"{'Improved':>12s} "
        f"{'Worse':>8s}"
    )

    raw = np.asarray(errors["raw"])

    for name in methods:
        values = np.asarray(errors[name])

        improved = int(np.sum(values < raw))
        worse = int(np.sum(values > raw))

        print(
            f"{name:14s} "
            f"{np.mean(values):12.4f} "
            f"{improved:12d} "
            f"{worse:8d}"
        )

    print("\nFIVE PREVIOUSLY WORST WINDOWS")

    for index in WORST_INDICES:
        if index not in per_window:
            raise ValueError(
                f"Expected baseline window missing: {index}"
            )

        row = per_window[index]

        print(
            f"Window {index}: "
            f"raw={row['raw']:.3f} | "
            f"butterworth={row['butterworth']:.3f} | "
            f"linear={row['linear']:.3f} | "
            f"median={row['median']:.3f}"
        )

    print("\nEVALUATION COMPLETED")
    print(
        "No cleaning method has been approved "
        "for clinical or automatic downstream use."
    )


if __name__ == "__main__":
    main()
