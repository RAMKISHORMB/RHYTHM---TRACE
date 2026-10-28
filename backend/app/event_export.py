
"""
TRACE — Event Evidence Export API
MEDAITHON'26 | Cardiology PS-001

Exports experimental ECG event evidence as JSON.

Uses existing event detection results.
Does not modify the ECG, event classifications,
or reference annotations.

Research prototype — not for clinical use.
"""

import json
import csv
import io
import math
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.event_api import load_demo_events


router = APIRouter(
    prefix="/api/events",
    tags=["TRACE Event Evidence Export"],
)


def find_event(event_id: str) -> dict:
    """
    Retrieve an existing event evidence package.
    """

    try:
        result = load_demo_events()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Research ECG data unavailable.",
        ) from exc

    for event in result["events"]:
        if event["event_id"] == event_id:
            return event

    raise HTTPException(
        status_code=404,
        detail="Event not found.",
    )


@router.get("/{event_id}/export/json")
def export_event_json(event_id: str):
    """
    Export a single event with ECG samples,
    timing measurements, and quality context.
    """

    event = find_event(event_id)

    document = {
        "export_format": "TRACE_EVENT_EVIDENCE_V1",
        "exported_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "recording_id": "MITBIH-100",
        "data_source": (
            "MIT-BIH Arrhythmia Database "
            "Record 100"
        ),
        "recording_type": "offline_research_ecg",
        "motion_available": event["motion_available"],
        "event": event,
        "research_only": True,
        "clinical_validated": False,
        "disclaimer": (
            "Experimental RR-timing candidate. "
            "Not a medical diagnosis or clinically "
            "validated assessment."
        ),
    }

    payload = json.dumps(
        document,
        indent=2,
        allow_nan=False,
    )

    filename = f"{event_id}.json"

    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Cache-Control": "no-store",
        },
    )

@router.get("/{event_id}/export/csv")
def export_event_csv(event_id: str):
    """
    Export the ECG snippet around one transient
    event candidate as a CSV file.

    Timestamps are relative to the beginning
    of the source ECG recording.

    ECG values are uncalibrated digital values.
    """

    event = find_event(event_id)
    evidence = event["evidence"]

    samples = evidence["ecg_samples"]
    fs = float(evidence["snippet_sampling_rate"])
    start = float(evidence["snippet_start_seconds"])
    event_time = float(evidence["timestamp_seconds"])

    if not math.isfinite(fs) or fs <= 0:
        raise HTTPException(
            status_code=503,
            detail="Invalid evidence sampling rate.",
        )

    if not math.isfinite(start) or not math.isfinite(event_time):
        raise HTTPException(
            status_code=503,
            detail="Invalid evidence timestamps.",
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow([
        "sample_index",
        "timestamp_seconds",
        "ecg_digital_value",
        "event_marker",
    ])

    # Event timestamp and snippet start are aligned
    # to the source recording's sampling grid.
    event_index = round((event_time - start) * fs)

    for index, value in enumerate(samples):
        timestamp = start + index / fs

        writer.writerow([
            index,
            f"{timestamp:.6f}",
            value,
            1 if index == event_index else 0,
        ])

    filename = f"{event_id}.csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Cache-Control": "no-store",
        },
    )
