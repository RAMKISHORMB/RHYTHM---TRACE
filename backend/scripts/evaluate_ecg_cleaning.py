
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window


DATA_FILE = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(a, dtype=np.float64)
                 - np.asarray(b, dtype=np.float64)) ** 2
            )
        )
    )


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    with np.load(DATA_FILE, allow_pickle=False) as data:
        references = data["ecg_reference"]
        corrupted = data["ecg_corrupted"]
        labels = data["labels"]
        class_names = data["class_names"]
        fs = int(data["fs"])

    print("TRACE — ECG CLEANING DATASET EVALUATION")
    print("---------------------------------------")
    print(f"Sampling rate: {fs} Hz")
    print(f"Total windows: {len(labels)}")

    assert len(references) == len(corrupted) == len(labels)

    class_names = [
        str(name) for name in class_names
    ]

    baseline_indices = [
        index
        for index, label in enumerate(labels)
        if (
            class_names[int(label)]
            if isinstance(label, (int, np.integer))
            else str(label)
        ) == "baseline_wander"
    ]

    if not baseline_indices:
        raise RuntimeError(
            "No baseline_wander windows found."
        )

    raw_errors = []
    cleaned_errors = []
    improved_count = 0

    for index in baseline_indices:
        reference = np.asarray(
            references[index], dtype=np.float64
        )
        noisy = np.asarray(
            corrupted[index], dtype=np.float64
        )

        result = clean_ecg_window(
            ecg_window=noisy,
            fs=fs,
            predicted_artifact="baseline_wander",
            trust_level="REVIEW",
        )

        if (
            result.status != "CANDIDATE_FILTERED"
            or result.cleaned_ecg is None
        ):
            raise AssertionError(
                f"Filtering failed for window {index}"
            )

        raw_error = rmse(noisy, reference)
        cleaned_error = rmse(
            result.cleaned_ecg, reference
        )

        raw_errors.append(raw_error)
        cleaned_errors.append(cleaned_error)

        if cleaned_error < raw_error:
            improved_count += 1

    mean_raw = float(np.mean(raw_errors))
    mean_cleaned = float(np.mean(cleaned_errors))

    reduction = (
        (mean_raw - mean_cleaned) / mean_raw * 100
        if mean_raw > 0
        else 0.0
    )

    print()
    print(f"Baseline windows: {len(baseline_indices)}")
    print(f"Mean raw RMSE: {mean_raw:.4f}")
    print(f"Mean cleaned RMSE: {mean_cleaned:.4f}")
    print(f"Mean RMSE reduction: {reduction:.2f}%")
    print(
        "Windows improved: "
        f"{improved_count}/{len(baseline_indices)}"
    )

    assert np.isfinite(mean_raw)
    assert np.isfinite(mean_cleaned)

    print()
    print("EVALUATION COMPLETED SUCCESSFULLY")

    if mean_cleaned < mean_raw:
        print("RESULT: Mean RMSE improved.")
    else:
        print(
            "RESULT: Mean RMSE did not improve. "
            "Filter requires investigation."
        )


if __name__ == "__main__":
    main()
