
"""
TRACE — Transient Event API
MEDAITHON'26 | Cardiology PS-001

Demo API using MIT-BIH Record 100.

Reference annotations are not used during
event detection or API response generation.

Research prototype only.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from app.transient_event_detector import detect_transient_events
from app.event_quality_context import create_quality_aware_event


router = APIRouter(
    prefix="/api/events",
    tags=["TRACE Cardiac Events"],
)

ROOT = Path(__file__).resolve().parents[1]
ECG_PATH = ROOT / "data" / "processed" / "100_250hz.npz"


@lru_cache(maxsize=1)
def load_demo_events():
    """
    Analyze a fixed five-minute research recording.

    Cached so repeated dashboard requests do not
    rerun the ECG detector unnecessarily.
    """

    if not ECG_PATH.exists():
        raise FileNotFoundError(
            f"Demo ECG recording not found: {ECG_PATH}"
        )

    with np.load(ECG_PATH, allow_pickle=False) as data:
        recording = np.asarray(data["ecg"])
        fs = int(data["fs"])

    if recording.ndim == 2:
        ecg = recording[:, 0]
    elif recording.ndim == 1:
        ecg = recording
    else:
        raise ValueError("Invalid ECG recording shape.")

    duration_seconds = 300
    ecg = ecg[:duration_seconds * fs]

    detected = detect_transient_events(
        ecg,
        fs=fs,
    )

    events = []

    for event in detected.events:
        package = create_quality_aware_event(
            ecg=ecg,
            event=event,
            fs=fs,
            imu=None,
            recording_id="MITBIH-100",
        )

        events.append(package.to_dict())

    return {
        "recording_id": "MITBIH-100",
        "recording_type": "offline_research_ecg",
        "duration_seconds": len(ecg) / fs,
        "sampling_rate": fs,
        "peak_count": detected.peak_count,
        "estimated_heart_rate_bpm": (
            detected.estimated_heart_rate_bpm
        ),
        "event_count": len(events),
        "events": events,
        "motion_available": False,
        "live_monitoring": False,
        "experimental": True,
        "clinical_validated": False,
        "disclaimer": (
            "Research event candidates only. "
            "Not diagnostic and not validated "
            "for clinical decision-making."
        ),
    }


@router.get("")
def list_events():
    try:
        return load_demo_events()

    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@router.get("/{event_id}")
def get_event(event_id: str):
    try:
        result = load_demo_events()

    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    for event in result["events"]:
        if event["event_id"] == event_id:
            return event

    raise HTTPException(
        status_code=404,
        detail="Event not found.",
    )
