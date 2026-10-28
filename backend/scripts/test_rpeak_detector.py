
"""
TRACE — R-Peak Detector Smoke Test

Tests candidate peak detection on MIT-BIH Record 100.

Ground-truth annotations are not passed to the
detector. Evaluation comes in a later stage.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rpeak_detector import detect_rpeaks


def main():
    path = (
        ROOT / "data" / "processed"
        / "100_250hz.npz"
    )

    with np.load(path, allow_pickle=False) as data:
        print("Available dataset keys:", data.files)

        # The recording was previously saved as
        # a two-channel ECG array.
        possible_keys = (
            "ecg",
            "signal",
            "signals",
            "ecg_resampled",
        )

        key = next(
            (k for k in possible_keys if k in data.files),
            None,
        )

        if key is None:
            raise KeyError(
                "Could not identify ECG array. "
                "Check the printed dataset keys."
            )

        recording = np.asarray(data[key])

    if recording.ndim == 2:
        ecg = recording[:, 0]
    else:
        ecg = recording

    fs = 250
    duration_seconds = 60
    window = ecg[:fs * duration_seconds]

    result = detect_rpeaks(window, fs=fs)

    print("\nTRACE — R-PEAK DETECTOR")
    print("=" * 55)
    print("Duration:", len(window) / fs, "seconds")
    print("Detected peaks:", result.peak_count)
    print(
        "Estimated heart rate:",
        result.estimated_heart_rate_bpm,
        "BPM",
    )
    print(
        "First 10 peak times:",
        result.peak_times_seconds[:10],
    )
    print(
        "First 10 RR intervals:",
        result.rr_intervals_seconds[:10],
    )
    print("\nSmoke test complete.")
    print(
        "Peak accuracy has not yet been evaluated."
    )


if __name__ == "__main__":
    main()
