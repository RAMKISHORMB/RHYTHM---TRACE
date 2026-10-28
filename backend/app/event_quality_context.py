
"""
TRACE — Event Quality Context
MEDAITHON'26 | Cardiology PS-001

Attaches experimental ECG quality evidence to
a transient cardiac-event evidence package.

This module does not classify cardiac rhythms,
establish clinical signal reliability, or
suppress event candidates.
"""

from dataclasses import dataclass, asdict

import numpy as np

from app.ecg_quality import analyze_ecg_quality
from app.event_evidence import create_event_evidence


@dataclass
class EventQualityResult:
    event_id: str
    event_type: str
    timestamp_seconds: float
    quality_status: str
    quality_flags: list[str]
    motion_available: bool
    evidence: dict
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def create_quality_aware_event(
    ecg,
    event,
    fs=250,
    imu=None,
    recording_id="unknown",
):
    """
    Create an ECG evidence package and attach
    quality analysis of the extracted snippet.

    The existing ECG quality detector is
    heuristic and may produce false warnings.
    """

    evidence = create_event_evidence(
        ecg=ecg,
        event=event,
        fs=fs,
        imu=imu,
        recording_id=recording_id,
    )

    snippet = np.asarray(
        evidence.ecg_samples,
        dtype=np.float64,
    )

    quality = analyze_ecg_quality(
        snippet,
        fs=fs,
    )

    flags = list(quality.quality_flags)

    if flags:
        status = "REVIEW_QUALITY"
    else:
        status = "NO_HEURISTIC_WARNING"

    context = {
        "status": status,
        "flags": flags,
        "method": "experimental_ecg_quality_heuristics",
        "clinical_validated": False,
        "note": (
            "Quality flags are provisional. "
            "No warning does not establish "
            "clinical signal reliability."
        ),
    }

    evidence.quality_context = context

    return EventQualityResult(
        event_id=evidence.event_id,
        event_type=evidence.event_type,
        timestamp_seconds=evidence.timestamp_seconds,
        quality_status=status,
        quality_flags=flags,
        motion_available=(
            evidence.motion_samples is not None
        ),
        evidence=evidence.to_dict(),
    )
