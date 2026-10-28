
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "trace_artifacts.npz"
ANNOTATIONS = (
    ROOT / "data" / "processed" / "100_annotations_250hz.npz"
)

WINDOW_SECONDS = [0.5, 1.0, 1.5, 2.0]


def rmse(a, b):
    return float(np.sqrt(np.mean(
        (np.asarray(a, dtype=np.float64)
         - np.asarray(b, dtype=np.float64)) ** 2
    )))


def clean(signal, fs, seconds):
    size = int(round(fs * seconds))
    if size % 2 == 0:
        size += 1

    baseline = median_filter(
        signal, size=size, mode="reflect"
    )
    return signal - baseline + np.median(signal)


def main():
    with np.load(DATA, allow_pickle=False) as d:
        reference = d["ecg_reference"]
        corrupted = d["ecg_corrupted"]
        labels = d["labels"]
        names = [str(x) for x in d["class_names"]]
        starts = d["start_samples"]
        fs = int(d["fs"])

    with np.load(ANNOTATIONS, allow_pickle=False) as d:
        beats = np.asarray(d["samples"], dtype=np.int64)
        assert int(d["fs"]) == fs

    indices = [
        i for i, label in enumerate(labels)
        if (
            names[int(label)]
            if isinstance(label, (int, np.integer))
            else str(label)
        ) == "baseline_wander"
    ]

    radius = round(0.10 * fs)

    print("TRACE — MEDIAN WINDOW SENSITIVITY")
    print("---------------------------------")
    print(f"Baseline windows: {len(indices)}")
    print("Candidate durations:", WINDOW_SECONDS)
    print()

    for seconds in WINDOW_SECONDS:
        raw_errors = []
        cleaned_errors = []
        beat_errors = []
        amplitude_errors = []
        raw_amplitude_errors = []
        worsened = 0
        beat_count = 0

        for i in indices:
            ref = np.asarray(reference[i], dtype=np.float64)
            noisy = np.asarray(corrupted[i], dtype=np.float64)
            candidate = clean(noisy, fs, seconds)

            raw_error = rmse(noisy, ref)
            filtered_error = rmse(candidate, ref)

            raw_errors.append(raw_error)
            cleaned_errors.append(filtered_error)

            if filtered_error > raw_error:
                worsened += 1

            start = int(starts[i])
            relevant = beats[
                (beats >= start) &
                (beats < start + len(ref))
            ]

            for absolute in relevant:
                center = int(absolute - start)
                left = center - radius
                right = center + radius + 1

                if left < 0 or right > len(ref):
                    continue

                r = ref[left:right]
                n = noisy[left:right]
                c = candidate[left:right]

                ref_ptp = float(np.ptp(r))
                if ref_ptp <= 1e-9:
                    continue

                beat_errors.append(rmse(c, r))

                raw_amplitude_errors.append(
                    abs(float(np.ptp(n)) - ref_ptp)
                    / ref_ptp * 100
                )
                amplitude_errors.append(
                    abs(float(np.ptp(c)) - ref_ptp)
                    / ref_ptp * 100
                )
                beat_count += 1

        print(f"MEDIAN WINDOW: {seconds:.1f} seconds")
        print(
            f"  Mean raw RMSE:       "
            f"{np.mean(raw_errors):.4f}"
        )
        print(
            f"  Mean cleaned RMSE:   "
            f"{np.mean(cleaned_errors):.4f}"
        )
        print(
            f"  Local beat RMSE:     "
            f"{np.mean(beat_errors):.4f}"
        )
        print(
            f"  Raw amplitude error: "
            f"{np.mean(raw_amplitude_errors):.3f}%"
        )
        print(
            f"  Clean amplitude err: "
            f"{np.mean(amplitude_errors):.3f}%"
        )
        print(f"  Worsened windows:    {worsened}")
        print(f"  Beat neighborhoods:  {beat_count}")
        print()

    print("SENSITIVITY ANALYSIS COMPLETED")
    print(
        "Exploratory comparison only; no filter "
        "approved for clinical use."
    )


if __name__ == "__main__":
    main()
