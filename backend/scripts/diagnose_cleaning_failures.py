
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window

DATA_FILE = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)

WORST_INDICES = [866, 2191, 1896, 1391, 1901]


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def centered_rmse(signal, reference):
    # Diagnostic only: removes each signal's mean.
    # This metric cannot detect constant-offset errors.
    return rmse(
        signal - np.mean(signal),
        reference - np.mean(reference),
    )


def main():
    with np.load(DATA_FILE, allow_pickle=False) as data:
        references = data["ecg_reference"]
        corrupted = data["ecg_corrupted"]
        labels = data["labels"]
        names = [str(x) for x in data["class_names"]]
        fs = int(data["fs"])

    edge = int(round(0.5 * fs))

    print("TRACE — CLEANING FAILURE DIAGNOSIS")
    print("----------------------------------")

    for index in WORST_INDICES:
        if not 0 <= index < len(labels):
            raise IndexError(f"Invalid window: {index}")

        label = labels[index]
        name = (
            names[int(label)]
            if isinstance(label, (int, np.integer))
            else str(label)
        )

        if name != "baseline_wander":
            raise ValueError(
                f"Window {index} is {name}, not baseline_wander"
            )

        reference = np.asarray(
            references[index], dtype=np.float64
        )
        noisy = np.asarray(
            corrupted[index], dtype=np.float64
        )

        result = clean_ecg_window(
            noisy, fs, "baseline_wander", "REVIEW"
        )

        if result.cleaned_ecg is None:
            raise RuntimeError(
                f"No cleaned output for window {index}"
            )

        cleaned = result.cleaned_ecg

        if len(reference) <= 2 * edge:
            raise ValueError("Window too short for edge analysis")

        print(f"\nWINDOW {index}")
        print(f"Reference mean: {np.mean(reference):.3f}")
        print(f"Corrupted mean: {np.mean(noisy):.3f}")
        print(f"Cleaned mean:   {np.mean(cleaned):.3f}")

        print(
            f"Raw RMSE:       {rmse(noisy, reference):.3f}"
        )
        print(
            f"Cleaned RMSE:   {rmse(cleaned, reference):.3f}"
        )

        print(
            "Centered raw RMSE:     "
            f"{centered_rmse(noisy, reference):.3f}"
        )
        print(
            "Centered cleaned RMSE: "
            f"{centered_rmse(cleaned, reference):.3f}"
        )

        for label, segment in [
            ("First 0.5s", slice(0, edge)),
            ("Middle 3.0s", slice(edge, -edge)),
            ("Last 0.5s", slice(-edge, None)),
        ]:
            raw_error = rmse(
                noisy[segment], reference[segment]
            )
            clean_error = rmse(
                cleaned[segment], reference[segment]
            )

            print(
                f"{label:12s} | "
                f"Raw: {raw_error:8.3f} | "
                f"Cleaned: {clean_error:8.3f}"
            )

    print("\nDIAGNOSIS COMPLETED")


if __name__ == "__main__":
    main()
