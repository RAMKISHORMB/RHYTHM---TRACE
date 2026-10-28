
"""
TRACE — Explainable Event Evidence Recorder
MEDAITHON'26 | Cardiology PS-001

Creates reviewable ECG evidence packages for
transient timing-anomaly candidates.

No reference beat annotations are used.
No diagnostic classifications are produced.
"""

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class EventEvidence:
    event_id: str
    event_type: str
    timestamp_seconds: float
    rr_interval_seconds: float
    baseline_rr_seconds: float
    ratio_to_baseline: float
    explanation: str
    snippet_start_seconds: float
    snippet_end_seconds: float
    snippet_sampling_rate: float
    ecg_samples: list[float]
    motion_samples: list[float] | None
    motion_source: str | None
    quality_context: dict[str, Any]
    requires_review: bool = True
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def create_event_evidence(
    ecg,
    event,
    fs=250,
    imu=None,
    pre_seconds=2.0,
    post_seconds=2.0,
    quality_context=None,
    recording_id="unknown",
    motion_source="simulated_imu",
):
    """
    Extract an ECG snippet around a detected event.

    ECG and IMU must already be time-aligned and
    share the same sampling rate.

    Timestamps are relative to the supplied ECG
    recording, not wall-clock time.
    """

    signal = np.asarray(ecg, dtype=np.float64)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("ECG must be a nonempty 1D array.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate.")

    if (
        not np.isfinite(pre_seconds)
        or not np.isfinite(post_seconds)
        or pre_seconds < 0
        or post_seconds < 0
    ):
        raise ValueError("Invalid snippet duration.")

    required_fields = (
        "event_type",
        "timestamp_seconds",
        "rr_interval_seconds",
        "baseline_rr_seconds",
        "ratio_to_baseline",
        "explanation",
    )

    if not isinstance(event, dict):
        raise ValueError("Event must be a dictionary.")

    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing event field: {field}")

    event_time = float(event["timestamp_seconds"])

    if (
        not np.isfinite(event_time)
        or event_time < 0
        or event_time >= signal.size / fs
    ):
        raise ValueError("Event timestamp outside recording.")

    motion = None

    if imu is not None:
        motion = np.asarray(imu, dtype=np.float64)

        if motion.shape != signal.shape:
            raise ValueError(
                "IMU must be a synchronized 1D array "
                "with the same length as ECG."
            )

        if not np.all(np.isfinite(motion)):
            raise ValueError("IMU contains invalid values.")

    event_sample = int(round(event_time * fs))

    start_sample = max(
        0,
        event_sample - int(round(pre_seconds * fs)),
    )

    end_sample = min(
        signal.size,
        event_sample + int(round(post_seconds * fs)),
    )

    if end_sample <= start_sample:
        raise ValueError("Empty evidence snippet.")

    ecg_snippet = signal[start_sample:end_sample]

    motion_snippet = (
        motion[start_sample:end_sample]
        if motion is not None
        else None
    )

    # Deterministic identifier for a recording
    # and event sample. No patient identifiers.
    event_id = (
        f"{recording_id}-"
        f"{event_sample}-"
        f"{event['event_type']}"
    )

    return EventEvidence(
        event_id=event_id,
        event_type=str(event["event_type"]),
        timestamp_seconds=event_time,
        rr_interval_seconds=float(
            event["rr_interval_seconds"]
        ),
        baseline_rr_seconds=float(
            event["baseline_rr_seconds"]
        ),
        ratio_to_baseline=float(
            event["ratio_to_baseline"]
        ),
        explanation=str(event["explanation"]),
        snippet_start_seconds=start_sample / fs,
        snippet_end_seconds=end_sample / fs,
        snippet_sampling_rate=float(fs),
        ecg_samples=ecg_snippet.tolist(),
        motion_samples=(
            motion_snippet.tolist()
            if motion_snippet is not None
            else None
        ),
        motion_source=(
            motion_source
            if motion_snippet is not None
            else None
        ),
        quality_context=(
            dict(quality_context)
            if quality_context is not None
            else {
                "status": "NOT_EVALUATED",
                "note": (
                    "No synchronized signal-quality "
                    "assessment supplied."
                ),
            }
        ),
    )
