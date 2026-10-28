
"""
TRACE — Transient Event Candidate Smoke Test

Runs on MIT-BIH Record 100 without supplying
reference beat annotations to the detector.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.transient_event_detector import (
    detect_transient_events,
)


def main():
    path = (
        ROOT / "data" / "processed"
        / "100_250hz.npz"
    )

    with np.load(path, allow_pickle=False) as data:
        recording = data["ecg"]
        fs = int(data["fs"])

    ecg = (
        recording[:, 0]
        if recording.ndim == 2
        else recording
    )

    # First evaluate five minutes.
    duration_seconds = 300
    window = ecg[:duration_seconds * fs]

    result = detect_transient_events(
        window,
        fs=fs,
    )

    print("\nTRACE — TRANSIENT EVENT CANDIDATES")
    print("=" * 65)
    print("Duration:", len(window) / fs, "seconds")
    print("Detected peaks:", result.peak_count)
    print(
        "Estimated heart rate:",
        result.estimated_heart_rate_bpm,
        "BPM",
    )
    print("Candidate events:", result.event_count)

    print("\nFIRST 15 EVENT CANDIDATES")
    print("-" * 65)

    for event in result.events[:15]:
        print(
            f"{event['timestamp_seconds']:8.3f}s | "
            f"{event['event_type']:20s} | "
            f"RR={event['rr_interval_seconds']:.3f}s | "
            f"baseline={event['baseline_rr_seconds']:.3f}s"
        )

    print("\nSmoke test complete.")
    print(
        "Timing anomalies only; "
        "not diagnostic arrhythmia labels."
    )


if __name__ == "__main__":
    main()
