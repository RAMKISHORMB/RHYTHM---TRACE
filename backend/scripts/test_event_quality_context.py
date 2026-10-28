
"""
TRACE — Event Quality Context Test

Uses the first five minutes of MIT-BIH
Record 100.

No synthetic IMU is attached to this
real ECG recording.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.transient_event_detector import (
    detect_transient_events,
)
from app.event_quality_context import (
    create_quality_aware_event,
)


def main():
    path = ROOT / "data/processed/100_250hz.npz"

    with np.load(path, allow_pickle=False) as data:
        recording = data["ecg"]
        fs = int(data["fs"])

    ecg = (
        recording[:, 0]
        if recording.ndim == 2
        else recording
    )

    ecg = ecg[:300 * fs]

    result = detect_transient_events(
        ecg,
        fs=fs,
    )

    print("\nTRACE — EVENT QUALITY CONTEXT TEST")
    print("=" * 65)

    for event in result.events:
        package = create_quality_aware_event(
            ecg=ecg,
            event=event,
            fs=fs,
            imu=None,
            recording_id="MITBIH-100",
        )

        print()
        print("Event ID:", package.event_id)
        print(
            "Timestamp:",
            package.timestamp_seconds,
        )
        print(
            "Quality status:",
            package.quality_status,
        )
        print(
            "Quality flags:",
            package.quality_flags,
        )
        print(
            "Motion available:",
            package.motion_available,
        )
        print(
            "ECG samples:",
            len(package.evidence["ecg_samples"]),
        )
        print(
            "Requires review:",
            package.evidence["requires_review"],
        )

        assert package.quality_status in (
            "REVIEW_QUALITY",
            "NO_HEURISTIC_WARNING",
        )
        assert (
            package.evidence["quality_context"]["status"]
            == package.quality_status
        )
        assert package.motion_available is False
        assert package.evidence["requires_review"] is True
        assert len(package.evidence["ecg_samples"]) == 1000

    print()
    print("EVENT QUALITY CONTEXT TEST COMPLETE")
    print(
        "Experimental ECG quality evidence only."
    )


if __name__ == "__main__":
    main()
