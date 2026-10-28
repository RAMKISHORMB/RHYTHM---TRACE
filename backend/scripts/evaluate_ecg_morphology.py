
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_cleaning import clean_ecg_window


DATA_FILE = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)

ANNOTATION_FILE = (
    ROOT / "data" / "processed"
    / "100_annotations_250hz.npz"
)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return float(np.sqrt(np.mean((a - b) ** 2)))


def peak_to_peak(signal):
    return float(np.ptp(signal))


def class_name(label, names):
    if isinstance(label, (int, np.integer)):
        return names[int(label)]
    return str(label)


def main():
    with np.load(DATA_FILE, allow_pickle=False) as data:
        references = data["ecg_reference"]
        corrupted = data["ecg_corrupted"]
        labels = data["labels"]
        names = [str(x) for x in data["class_names"]]
        starts = data["start_samples"]
        fs = int(data["fs"])

    with np.load(
        ANNOTATION_FILE, allow_pickle=False
    ) as annotations:
        beat_samples = np.asarray(
            annotations["samples"], dtype=np.int64
        )
        annotation_fs = int(annotations["fs"])

    if fs != annotation_fs:
        raise ValueError(
            "Dataset and annotation sampling rates differ."
        )

    baseline_indices = [
        i for i, label in enumerate(labels)
        if class_name(label, names) == "baseline_wander"
    ]

    if not baseline_indices:
        raise RuntimeError("No baseline-wander windows.")

    radius = int(round(0.10 * fs))

    raw_errors = []
    cleaned_errors = []

    raw_local_errors = []
    cleaned_local_errors = []

    raw_amplitude_errors = []
    cleaned_amplitude_errors = []

    window_results = []
    beat_count = 0

    for index in baseline_indices:
        reference = np.asarray(
            references[index], dtype=np.float64
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

        assert result.status == "CANDIDATE_FILTERED"
        assert result.cleaned_ecg is not None

        cleaned = result.cleaned_ecg

        raw_error = rmse(noisy, reference)
        cleaned_error = rmse(cleaned, reference)

        raw_errors.append(raw_error)
        cleaned_errors.append(cleaned_error)

        window_results.append(
            (
                index,
                int(starts[index]),
                raw_error,
                cleaned_error,
                cleaned_error - raw_error,
            )
        )

        start = int(starts[index])
        end = start + len(reference)

        # These are reference annotations, not
        # predictions from an arrhythmia detector.
        relevant_beats = beat_samples[
            (beat_samples >= start)
            & (beat_samples < end)
        ]

        for absolute_sample in relevant_beats:
            center = int(absolute_sample - start)

            left = center - radius
            right = center + radius + 1

            # Exclude incomplete beat neighborhoods.
            if left < 0 or right > len(reference):
                continue

            ref_segment = reference[left:right]
            raw_segment = noisy[left:right]
            clean_segment = cleaned[left:right]

            raw_local_errors.append(
                rmse(raw_segment, ref_segment)
            )
            cleaned_local_errors.append(
                rmse(clean_segment, ref_segment)
            )

            ref_amplitude = peak_to_peak(ref_segment)

            if ref_amplitude <= 1e-9:
                continue

            raw_amplitude_errors.append(
                abs(
                    peak_to_peak(raw_segment)
                    - ref_amplitude
                ) / ref_amplitude * 100
            )

            cleaned_amplitude_errors.append(
                abs(
                    peak_to_peak(clean_segment)
                    - ref_amplitude
                ) / ref_amplitude * 100
            )

            beat_count += 1

    print("TRACE — ECG MORPHOLOGY EVALUATION")
    print("--------------------------------")
    print(f"Sampling rate: {fs} Hz")
    print(f"Baseline windows: {len(baseline_indices)}")
    print(f"Beat neighborhoods evaluated: {beat_count}")
    print("Local beat radius: +/- 100 ms")

    print()
    print("FULL-WINDOW RMSE")
    print(
        f"Raw:     {np.mean(raw_errors):.4f}"
    )
    print(
        f"Cleaned: {np.mean(cleaned_errors):.4f}"
    )

    print()
    print("LOCAL BEAT RMSE")
    print(
        f"Raw:     {np.mean(raw_local_errors):.4f}"
    )
    print(
        f"Cleaned: {np.mean(cleaned_local_errors):.4f}"
    )

    print()
    print("BEAT PEAK-TO-PEAK AMPLITUDE ERROR")
    print(
        f"Raw:     "
        f"{np.mean(raw_amplitude_errors):.2f}%"
    )
    print(
        f"Cleaned: "
        f"{np.mean(cleaned_amplitude_errors):.2f}%"
    )

    print()
    print("WORST FIVE WINDOWS AFTER CLEANING")
    print(
        "Index | Start sample | Raw RMSE | "
        "Cleaned RMSE | Change"
    )

    # Largest increase in RMSE comes first.
    worst = sorted(
        window_results,
        key=lambda row: row[4],
        reverse=True,
    )[:5]

    for index, start, raw, cleaned, change in worst:
        print(
            f"{index:5d} | "
            f"{start:12d} | "
            f"{raw:8.3f} | "
            f"{cleaned:12.3f} | "
            f"{change:+8.3f}"
        )

    print()
    print("EVALUATION COMPLETED")

    if (
        np.mean(cleaned_local_errors)
        < np.mean(raw_local_errors)
    ):
        print(
            "Local beat RMSE improved on average."
        )
    else:
        print(
            "WARNING: Local beat RMSE did not improve."
        )

    if (
        np.mean(cleaned_amplitude_errors)
        > np.mean(raw_amplitude_errors)
    ):
        print(
            "WARNING: Peak-to-peak amplitude "
            "error increased."
        )
    else:
        print(
            "Peak-to-peak amplitude error "
            "did not increase on average."
        )


if __name__ == "__main__":
    main()
