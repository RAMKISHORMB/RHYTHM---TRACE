
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "trace_artifacts.npz"
BEATS_FILE = ROOT / "data" / "processed" / "100_annotations_250hz.npz"


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def clean_with_median(ecg, fs):
    signal = np.asarray(ecg, dtype=np.float64)

    size = fs if fs % 2 else fs + 1

    baseline = median_filter(
        signal,
        size=size,
        mode="reflect",
    )

    return signal - baseline + np.median(signal)


def get_label(label, names):
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

    with np.load(BEATS_FILE, allow_pickle=False) as data:
        beat_samples = np.asarray(
            data["samples"], dtype=np.int64
        )
        beat_fs = int(data["fs"])

    assert fs == beat_fs

    indices = [
        i for i, label in enumerate(labels)
        if get_label(label, names) == "baseline_wander"
    ]

    radius = round(0.10 * fs)

    raw_window_errors = []
    cleaned_window_errors = []
    raw_beat_errors = []
    cleaned_beat_errors = []
    raw_amplitude_errors = []
    cleaned_amplitude_errors = []
    failures = []
    evaluated_beats = 0

    for index in indices:
        reference = np.asarray(
            references[index], dtype=np.float64
        )
        noisy = np.asarray(
            corrupted[index], dtype=np.float64
        )

        cleaned = clean_with_median(noisy, fs)

        assert cleaned.shape == reference.shape
        assert np.all(np.isfinite(cleaned))

        raw_error = rmse(noisy, reference)
        cleaned_error = rmse(cleaned, reference)

        raw_window_errors.append(raw_error)
        cleaned_window_errors.append(cleaned_error)

        if cleaned_error > raw_error:
            failures.append(
                (index, raw_error, cleaned_error)
            )

        start = int(starts[index])
        end = start + len(reference)

        relevant = beat_samples[
            (beat_samples >= start)
            & (beat_samples < end)
        ]

        for absolute_sample in relevant:
            center = int(absolute_sample - start)
            left = center - radius
            right = center + radius + 1

            if left < 0 or right > len(reference):
                continue

            ref = reference[left:right]
            raw = noisy[left:right]
            candidate = cleaned[left:right]

            ref_amplitude = float(np.ptp(ref))

            if ref_amplitude <= 1e-9:
                continue

            raw_beat_errors.append(rmse(raw, ref))
            cleaned_beat_errors.append(
                rmse(candidate, ref)
            )

            raw_amplitude_errors.append(
                abs(float(np.ptp(raw)) - ref_amplitude)
                / ref_amplitude * 100
            )

            cleaned_amplitude_errors.append(
                abs(float(np.ptp(candidate)) - ref_amplitude)
                / ref_amplitude * 100
            )

            evaluated_beats += 1

    raw_mean = float(np.mean(raw_window_errors))
    cleaned_mean = float(np.mean(cleaned_window_errors))

    print("TRACE — MEDIAN CLEANING VALIDATION")
    print("----------------------------------")
    print(f"Sampling rate: {fs} Hz")
    print(f"Baseline windows: {len(indices)}")
    print(f"Beat neighborhoods: {evaluated_beats}")

    print("\nFULL-WINDOW RMSE")
    print(f"Raw:     {raw_mean:.4f}")
    print(f"Cleaned: {cleaned_mean:.4f}")
    print(
        "Reduction: "
        f"{(raw_mean - cleaned_mean) / raw_mean * 100:.2f}%"
    )

    print("\nLOCAL BEAT RMSE (+/- 100 ms)")
    print(f"Raw:     {np.mean(raw_beat_errors):.4f}")
    print(f"Cleaned: {np.mean(cleaned_beat_errors):.4f}")

    print("\nBEAT PEAK-TO-PEAK AMPLITUDE ERROR")
    print(f"Raw:     {np.mean(raw_amplitude_errors):.3f}%")
    print(
        f"Cleaned: "
        f"{np.mean(cleaned_amplitude_errors):.3f}%"
    )

    print("\nWINDOWS THAT BECAME WORSE")
    print(f"Count: {len(failures)}")

    for index, raw, cleaned in sorted(
        failures,
        key=lambda row: row[2] - row[1],
        reverse=True,
    ):
        print(
            f"Window {index}: "
            f"raw={raw:.3f}, "
            f"cleaned={cleaned:.3f}, "
            f"increase={cleaned - raw:+.3f}"
        )

    print("\nVALIDATION COMPLETED")

    if (
        np.mean(cleaned_beat_errors)
        < np.mean(raw_beat_errors)
    ):
        print("PASS: Mean local beat RMSE improved.")
    else:
        print("WARNING: Mean local beat RMSE worsened.")

    if (
        np.mean(cleaned_amplitude_errors)
        <= np.mean(raw_amplitude_errors)
    ):
        print("PASS: Mean amplitude error did not increase.")
    else:
        print("WARNING: Mean amplitude error increased.")

    print(
        "Research-only evaluation. "
        "No clinical validation or safety approval."
    )


if __name__ == "__main__":
    main()
