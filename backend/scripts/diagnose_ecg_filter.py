
"""
TRACE — Stage 11.7.1
ECG Filter Diagnostic

Checks:
- Original ECG amplitude and offset
- Filtered ECG amplitude
- Signal finiteness
- Difference between original and filtered ECG
- Baseline-centered waveform distortion

Does not modify project data.
"""

import numpy as np

from app.corrupted_event_evaluation import (
    load_evaluation_ecg,
)

from app.motion_compensation_baseline import (
    apply_ecg_filter,
)


def summarize(name, signal):
    signal = np.asarray(signal, dtype=np.float64)

    print()
    print(name)
    print("-" * 50)
    print("Samples:", len(signal))
    print("Minimum:", float(np.min(signal)))
    print("Maximum:", float(np.max(signal)))
    print("Mean:", float(np.mean(signal)))
    print("Median:", float(np.median(signal)))
    print("Standard deviation:", float(np.std(signal)))
    print("All finite:", bool(np.isfinite(signal).all()))


def rms(signal):
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(signal ** 2)))


def main():
    print("=" * 60)
    print("TRACE — ECG FILTER DIAGNOSTIC")
    print("=" * 60)

    reference = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    filtered = apply_ecg_filter(reference)

    summarize("ORIGINAL ECG", reference)
    summarize("FILTERED ECG", filtered)

    original_centered = reference - np.median(reference)
    filtered_centered = filtered - np.median(filtered)

    print()
    print("COMPARISON")
    print("-" * 50)

    print(
        "Raw signal difference RMS:",
        rms(filtered - reference),
    )

    print(
        "Median-centered difference RMS:",
        rms(filtered_centered - original_centered),
    )

    print(
        "Original median:",
        float(np.median(reference)),
    )

    print(
        "Filtered median:",
        float(np.median(filtered)),
    )

    print(
        "Median offset difference:",
        float(
            np.median(reference)
            - np.median(filtered)
        ),
    )

    print()
    print("PASS: ECG filter diagnostic completed.")


if __name__ == "__main__":
    main()
