
"""
TRACE — R-Peak Detection Evaluation
MEDAITHON'26 | PS-001

Evaluates candidate R peaks against MIT-BIH
Record 100 reference beat annotations.

Uses one-to-one temporal matching within 150 ms.
Annotations are never supplied to the detector.

Research evaluation only; not clinical validation.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rpeak_detector import detect_rpeaks


# Standard WFDB beat annotation symbols.
# Excludes rhythm changes and non-beat markers.
BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F",
    "e", "j", "n", "E",
    "/", "f", "Q", "?",
}


def match_peaks(predicted, reference, tolerance_samples):
    """
    One-to-one chronological matching.

    Returns TP, FP, FN, and signed sample errors.
    Both arrays must be sorted.
    """
    predicted = np.asarray(predicted, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)

    i = 0
    j = 0
    errors = []

    while i < len(predicted) and j < len(reference):
        difference = int(predicted[i] - reference[j])

        if abs(difference) <= tolerance_samples:
            errors.append(difference)
            i += 1
            j += 1

        elif predicted[i] < reference[j] - tolerance_samples:
            i += 1

        else:
            j += 1

    tp = len(errors)
    fp = len(predicted) - tp
    fn = len(reference) - tp

    return tp, fp, fn, np.asarray(errors, dtype=np.int64)


def evaluate(duration_seconds):
    ecg_path = (
        ROOT / "data" / "processed" / "100_250hz.npz"
    )

    annotation_path = (
        ROOT / "data" / "processed"
        / "100_annotations_250hz.npz"
    )

    with np.load(ecg_path, allow_pickle=False) as data:
        recording = np.asarray(data["ecg"])
        fs = int(data["fs"])

    with np.load(annotation_path, allow_pickle=False) as data:
        annotation_samples = np.asarray(data["samples"])
        annotation_symbols = np.asarray(data["symbols"])
        annotation_fs = int(data["fs"])

    if fs != annotation_fs:
        raise ValueError("ECG and annotation sampling rates differ.")

    if recording.ndim == 2:
        ecg = recording[:, 0]
    elif recording.ndim == 1:
        ecg = recording
    else:
        raise ValueError("Unexpected ECG dimensions.")

    end_sample = min(
        len(ecg),
        int(round(duration_seconds * fs)),
    )

    if end_sample < 4 * fs:
        raise ValueError("Insufficient ECG duration.")

    # Use only reference beat annotations.
    beat_mask = np.isin(
        annotation_symbols,
        list(BEAT_SYMBOLS),
    )

    beat_samples = annotation_samples[beat_mask]

    # Exclude the first/last 0.2 seconds to
    # reduce evaluation edge effects.
    edge = int(round(0.2 * fs))

    reference = beat_samples[
        (beat_samples >= edge)
        & (beat_samples < end_sample - edge)
    ]

    result = detect_rpeaks(
        ecg[:end_sample],
        fs=fs,
    )

    predicted = np.asarray(
        result.peak_indices,
        dtype=np.int64,
    )

    predicted = predicted[
        (predicted >= edge)
        & (predicted < end_sample - edge)
    ]

    tolerance_ms = 150.0
    tolerance_samples = int(
        round(tolerance_ms * fs / 1000)
    )

    tp, fp, fn, errors = match_peaks(
        predicted,
        reference,
        tolerance_samples,
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    absolute_errors_ms = (
        np.abs(errors) * 1000.0 / fs
    )

    print("\nTRACE — R-PEAK ACCURACY EVALUATION")
    print("=" * 58)
    print("Record: MIT-BIH 100")
    print("Duration:", end_sample / fs, "seconds")
    print("Sampling frequency:", fs, "Hz")
    print("Matching tolerance:", tolerance_ms, "ms")
    print("Reference beats:", len(reference))
    print("Detected peaks:", len(predicted))
    print()
    print("True positives:", tp)
    print("False positives:", fp)
    print("False negatives:", fn)
    print()
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 score:  {f1 * 100:.2f}%")

    if len(absolute_errors_ms):
        print()
        print(
            "Median absolute timing error:",
            f"{np.median(absolute_errors_ms):.2f} ms",
        )
        print(
            "95th percentile timing error:",
            f"{np.percentile(absolute_errors_ms, 95):.2f} ms",
        )

    print()
    print("Evaluation complete.")
    print("Single record; no clinical validation.")


def main():
    # Evaluate the complete available recording.
    evaluate(duration_seconds=1800)


if __name__ == "__main__":
    main()
