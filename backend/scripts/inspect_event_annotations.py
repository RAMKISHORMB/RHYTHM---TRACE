
"""
TRACE — Transient Event Annotation Inspection
MEDAITHON'26 | PS-001

Matches detected RR event candidates to the nearest
MIT-BIH reference beat for offline evaluation.

Annotations are NOT used during event inference.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.transient_event_detector import detect_transient_events


def main():
    ecg_path = ROOT / "data/processed/100_250hz.npz"
    ann_path = (
        ROOT / "data/processed/100_annotations_250hz.npz"
    )

    with np.load(ecg_path, allow_pickle=False) as data:
        recording = data["ecg"]
        fs = int(data["fs"])

    with np.load(ann_path, allow_pickle=False) as data:
        samples = data["samples"].copy()
        symbols = data["symbols"].copy()
        ann_fs = int(data["fs"])

    if fs != ann_fs:
        raise ValueError("Sampling rates do not match.")

    ecg = (
        recording[:, 0]
        if recording.ndim == 2
        else recording
    )

    duration_seconds = 300

    result = detect_transient_events(
        ecg[:duration_seconds * fs],
        fs=fs,
    )

    # Include only reference beat annotations.
    beat_symbols = {
        "N", "L", "R", "B",
        "A", "a", "J", "S",
        "V", "r", "F",
        "e", "j", "n", "E",
        "/", "f", "Q", "?",
    }

    beat_mask = np.isin(symbols, list(beat_symbols))
    samples = samples[beat_mask]
    symbols = symbols[beat_mask]

    print("\nTRACE — EVENT ANNOTATION INSPECTION")
    print("=" * 75)

    for event in result.events:
        event_time = event["timestamp_seconds"]
        event_sample = int(round(event_time * fs))

        nearest_index = int(
            np.argmin(np.abs(samples - event_sample))
        )

        reference_sample = int(samples[nearest_index])
        reference_symbol = str(symbols[nearest_index])

        difference_ms = (
            abs(reference_sample - event_sample)
            * 1000.0 / fs
        )

        # Report surrounding beat annotations.
        start = max(0, nearest_index - 2)
        stop = min(len(samples), nearest_index + 3)

        neighbors = [
            (
                round(float(samples[i] / fs), 3),
                str(symbols[i]),
            )
            for i in range(start, stop)
        ]

        print()
        print("Event time:", round(event_time, 3), "s")
        print("Event type:", event["event_type"])
        print("Nearest annotation:", reference_symbol)
        print(
            "Annotation timing difference:",
            round(difference_ms, 2),
            "ms",
        )
        print("Surrounding beats:", neighbors)

        if difference_ms > 150:
            print(
                "WARNING: No reference beat within "
                "the 150 ms matching tolerance."
            )

    print("\nInspection complete.")
    print(
        "Annotation symbols describe reference beats, "
        "not diagnoses made by TRACE."
    )


if __name__ == "__main__":
    main()
