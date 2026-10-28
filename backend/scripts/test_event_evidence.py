
"""
TRACE — Event Evidence Recorder Test

Uses MIT-BIH Record 100 and the existing
transient event detector.

The record contains ECG but no synchronized
real-world IMU. Motion context is omitted.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.transient_event_detector import (
    detect_transient_events,
)
from app.event_evidence import (
    create_event_evidence,
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

    # First five minutes.
    ecg = ecg[:300 * fs]

    result = detect_transient_events(
        ecg,
        fs=fs,
    )

    print("\nTRACE — EVENT EVIDENCE TEST")
    print("=" * 65)

    for event in result.events:
        evidence = create_event_evidence(
            ecg=ecg,
            event=event,
            fs=fs,
            imu=None,
            recording_id="MITBIH-100",
        )

        print()
        print("Event ID:", evidence.event_id)
        print("Event type:", evidence.event_type)
        print("Timestamp:", evidence.timestamp_seconds)
        print(
            "ECG snippet:",
            evidence.snippet_start_seconds,
            "to",
            evidence.snippet_end_seconds,
            "seconds",
        )
        print(
            "ECG sample count:",
            len(evidence.ecg_samples),
        )
        print(
            "Motion available:",
            evidence.motion_samples is not None,
        )
        print(
            "Quality context:",
            evidence.quality_context["status"],
        )
        print("Requires review:", evidence.requires_review)

        assert len(evidence.ecg_samples) > 0
        assert evidence.requires_review is True
        assert evidence.clinical_validated is False

    print()
    print("EVENT EVIDENCE TEST COMPLETE")
    print(
        "Research evidence packages only; "
        "not clinical diagnostic reports."
    )


if __name__ == "__main__":
    main()
