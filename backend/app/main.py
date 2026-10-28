from app.trace_analysis_api import router as trace_analysis_router

"""
TRACE — FastAPI Backend
Trust-Routed Artefact-Classified ECG

MEDAITHON'26 | Cardiology PS-001

Research prototype features:
- Health and dataset readiness checks
- Recorded MIT-BIH ECG replay WebSocket
- Experimental artefact simulation WebSocket
- Transient cardiac-event evidence API
- JSON and CSV evidence export
- Persistent event-review audit trail

Not intended for clinical diagnosis or patient monitoring.
"""

import asyncio
from pathlib import Path

import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from app.artifact_stream import router as artifact_router
from app.event_api import router as event_router
from app.event_export import router as export_router
from app.event_audit import router as audit_router


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ECG_PATH = ROOT / "data" / "processed" / "100_250hz.npz"

ANNOTATION_PATH = (
    ROOT / "data" / "processed" / "100_annotations_250hz.npz"
)

ARTIFACT_DATASET_PATH = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)

ARTIFACT_MODEL_PATH = (
    ROOT / "models" / "artifact_classifier.joblib"
)

AUDIT_DATABASE_PATH = (
    ROOT / "data" / "trace_event_audit.db"
)

EXPECTED_FS = 250
FRAME_SIZE = 25
FRAME_DELAY = 0.1


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TRACE API",
    description=(
        "Research-only ECG replay, experimental "
        "artefact classification, explainable "
        "transient-event detection, evidence "
        "export, and event-review audit history."
    ),
    version="0.10.3",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTERS
# ============================================================

# Experimental artefact simulation WebSocket.
app.include_router(artifact_router)

# Cardiac-event evidence retrieval.
app.include_router(event_router)

# JSON and CSV evidence export.
app.include_router(export_router)

# SQLite-backed event-review history.
app.include_router(audit_router)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """
    Return project information and available endpoints.
    """

    return {
        "project": "TRACE",
        "full_name": "Trust-Routed Artefact-Classified ECG",
        "status": "running",
        "version": app.version,
        "research_only": True,
        "clinical_validated": False,
        "live_patient_monitoring": False,
        "endpoints": {
            "health": "/health",
            "api_documentation": "/docs",
            "recorded_ecg": "/ws/ecg",
            "artifact_demo": "/ws/artifact-demo",
            "cardiac_events": "/api/events",
            "cardiac_event_detail": "/api/events/{event_id}",
            "event_json_export": (
                "/api/events/{event_id}/export/json"
            ),
            "event_csv_export": (
                "/api/events/{event_id}/export/csv"
            ),
            "submit_event_review": "/api/event-audit/review",
            "event_review_history": (
                "/api/event-audit/{event_id}"
            ),
            "unified_window": "/api/trace/unified-window",
            "event_analysis": "/api/trace/event-analysis",
            "virtual_wearable": "/api/trace/virtual-wearable",
            "experiments": "/api/trace/experiments",
        },
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health():
    """
    Check availability of local research assets.
    """

    return {
        "status": "ok",
        "project": "TRACE",
        "research_only": True,
        "ecg_dataset_ready": ECG_PATH.exists(),
        "annotations_ready": ANNOTATION_PATH.exists(),
        "artifact_dataset_ready": (
            ARTIFACT_DATASET_PATH.exists()
        ),
        "artifact_model_ready": (
            ARTIFACT_MODEL_PATH.exists()
        ),
        "audit_database_initialized": (
            AUDIT_DATABASE_PATH.exists()
        ),
    }


# ============================================================
# RECORDED ECG REPLAY WEBSOCKET
# ============================================================

@app.websocket("/ws/ecg")
async def ecg_stream(websocket: WebSocket):
    """
    Replay MIT-BIH Record 100 at 250 Hz.

    ECG:
        Prerecorded research dataset.

    Beat symbols:
        Reference annotations, not AI predictions.

    IMU:
        Unavailable for this recording.

    Not a live wearable or clinical monitoring service.
    """

    await websocket.accept()

    try:
        # ----------------------------------------------------
        # VALIDATE INPUT FILES
        # ----------------------------------------------------

        if not ECG_PATH.exists():
            await websocket.send_json({
                "type": "error",
                "message": "Processed ECG file not found.",
            })
            return

        if not ANNOTATION_PATH.exists():
            await websocket.send_json({
                "type": "error",
                "message": "Processed annotation file not found.",
            })
            return

        # ----------------------------------------------------
        # LOAD RECORDED ECG
        # ----------------------------------------------------

        with np.load(
            ECG_PATH,
            allow_pickle=False,
        ) as data:
            ecg = data["ecg"].copy()
            fs = int(data["fs"])
            record_id = str(data["record_id"])
            units = str(data["units"])

        if fs != EXPECTED_FS:
            raise ValueError(
                f"Expected {EXPECTED_FS} Hz, received {fs} Hz."
            )

        if ecg.ndim != 2 or ecg.shape[1] < 1:
            raise ValueError(
                f"Unexpected ECG shape: {ecg.shape}"
            )

        signal = ecg[:, 0]

        if not np.isfinite(signal).all():
            raise ValueError(
                "ECG contains non-finite samples."
            )

        # ----------------------------------------------------
        # LOAD REFERENCE BEAT ANNOTATIONS
        # ----------------------------------------------------

        with np.load(
            ANNOTATION_PATH,
            allow_pickle=False,
        ) as data:
            beat_samples = data["samples"].copy()
            beat_symbols = data["symbols"].copy()
            annotation_fs = int(data["fs"])

        if annotation_fs != fs:
            raise ValueError(
                "ECG and annotation sampling rates differ."
            )

        if len(beat_samples) != len(beat_symbols):
            raise ValueError(
                "Annotation samples and symbols mismatch."
            )

        # ----------------------------------------------------
        # STREAM ECG FRAMES
        # ----------------------------------------------------

        beat_index = 0

        for start in range(
            0,
            len(signal),
            FRAME_SIZE,
        ):
            end = min(
                start + FRAME_SIZE,
                len(signal),
            )

            frame_beats = []

            while (
                beat_index < len(beat_samples)
                and int(beat_samples[beat_index]) < end
            ):
                sample = int(beat_samples[beat_index])

                if sample >= start:
                    frame_beats.append({
                        "sample": sample,
                        "symbol": str(
                            beat_symbols[beat_index]
                        ),
                        "source": "dataset_annotation",
                    })

                beat_index += 1

            await websocket.send_json({
                "type": "ecg_frame",
                "t0_sample": start,
                "fs": fs,
                "simulated": False,
                "replayed": True,
                "source": "MIT-BIH Arrhythmia Database",
                "record_id": record_id,
                "ecg_units": units,
                "ecg_raw": signal[start:end].tolist(),
                "beats": frame_beats,
                "imu": None,
                "imu_simulated": False,
            })

            await asyncio.sleep(FRAME_DELAY)

        # ----------------------------------------------------
        # REPLAY COMPLETE
        # ----------------------------------------------------

        await websocket.send_json({
            "type": "replay_complete",
            "record_id": record_id,
            "samples_streamed": len(signal),
        })

    except WebSocketDisconnect:
        print("TRACE ECG replay client disconnected.")

    except Exception as exc:
        print("TRACE ECG replay error:", exc)

        try:
            await websocket.send_json({
                "type": "error",
                "message": "ECG replay failed.",
            })
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass

app.include_router(trace_analysis_router)
