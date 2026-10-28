
import asyncio
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.artifact_inference import ArtifactInference
from app.trust_engine import evaluate_trust
from app.beat_alignment import (
    align_window_beats,
    load_reference_annotations,
)
from app.audit_engine import create_audit_entry


# ============================================================
# TRACE — ARTEFACT + TRUST + BEAT + AUDIT STREAM
# Experimental research prototype | Not for clinical use
# ============================================================

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)

FRAME_SIZE = 25
FRAME_DELAY = 0.1
DEMO_WINDOWS = 5
EXPECTED_FS = 250


def determine_audit_action(trust_level: str) -> str:
    """
    Conservative policy for the current demo.

    HIGH:
        Preserve original ECG.

    REVIEW:
        Preserve original ECG; do not claim cleaning
        was performed.

    UNRELIABLE:
        Abstain from reconstruction/interpretation.
        Raw replay remains available for inspection.
    """

    if trust_level == "HIGH":
        return "PRESERVED"

    if trust_level == "REVIEW":
        return "PRESERVED"

    if trust_level == "UNRELIABLE":
        return "ABSTAINED"

    raise ValueError(
        f"Unsupported trust level: {trust_level}"
    )


@router.websocket("/ws/artifact-demo")
async def artifact_demo(websocket: WebSocket):
    await websocket.accept()

    try:
        # ----------------------------------------------------
        # 1. Check dataset
        # ----------------------------------------------------

        if not DATA_PATH.exists():
            await websocket.send_json({
                "type": "error",
                "message": "Artefact dataset not found.",
            })
            return

        # ----------------------------------------------------
        # 2. Load classifier and reference annotations
        # ----------------------------------------------------

        classifier = ArtifactInference()

        (
            annotation_samples,
            annotation_symbols,
            annotation_fs,
        ) = load_reference_annotations()

        # ----------------------------------------------------
        # 3. Load demonstration data
        # ----------------------------------------------------

        with np.load(
            DATA_PATH,
            allow_pickle=False,
        ) as data:
            ecg_reference = (
                data["ecg_reference"][:DEMO_WINDOWS].copy()
            )

            ecg_corrupted = (
                data["ecg_corrupted"][:DEMO_WINDOWS].copy()
            )

            imu_simulated = (
                data["imu_simulated"][:DEMO_WINDOWS].copy()
            )

            labels = (
                data["labels"][:DEMO_WINDOWS].copy()
            )

            start_samples = (
                data["start_samples"][:DEMO_WINDOWS].copy()
            )

            class_names = data["class_names"].tolist()
            fs = int(data["fs"])

        if fs != EXPECTED_FS or annotation_fs != fs:
            raise ValueError(
                "ECG and annotation sampling rates do not match."
            )

        window_count = len(labels)

        if not all(
            len(array) == window_count
            for array in (
                ecg_reference,
                ecg_corrupted,
                imu_simulated,
                start_samples,
            )
        ):
            raise ValueError(
                "Artefact dataset arrays have mismatched lengths."
            )

        # ----------------------------------------------------
        # 4. Process each experimental ECG window
        # ----------------------------------------------------

        for window_index in range(window_count):
            reference = ecg_reference[window_index]
            corrupted = ecg_corrupted[window_index]
            imu = imu_simulated[window_index]

            start_sample = int(
                start_samples[window_index]
            )

            if not (
                len(reference)
                == len(corrupted)
                == len(imu)
            ):
                raise ValueError(
                    "ECG and simulated IMU window lengths differ."
                )

            # ------------------------------------------------
            # 5. AI artefact classification
            # ------------------------------------------------

            prediction = classifier.predict(
                corrupted,
                imu,
            )

            predicted_class = prediction["predicted_class"]
            model_probability = prediction["model_probability"]

            true_class = class_names[
                int(labels[window_index])
            ]

            # ------------------------------------------------
            # 6. Window-level signal trust
            # ------------------------------------------------

            trust = evaluate_trust(
                ecg_window=corrupted,
                predicted_class=predicted_class,
                model_probability=model_probability,
            )

            # ------------------------------------------------
            # 7. Beat alignment + beat-level trust
            # ------------------------------------------------

            beats = align_window_beats(
                ecg_window=corrupted,
                window_start_sample=start_sample,
                annotation_samples=annotation_samples,
                annotation_symbols=annotation_symbols,
                fs=fs,
                window_trust=trust.level,
            )

            # ------------------------------------------------
            # 8. Create the audit decision
            #
            # The audit entry reflects the action actually
            # authorized by the current streaming policy.
            # No ECG cleaning is performed here.
            # ------------------------------------------------

            action = determine_audit_action(trust.level)

            audit = create_audit_entry(
                window_index=window_index,
                artifact_class=predicted_class,
                trust_level=trust.level,
                action=action,
            )

            # ------------------------------------------------
            # 9. Send existing prediction event
            #
            # All original frontend fields are preserved.
            # ------------------------------------------------

            await websocket.send_json({
                "type": "artifact_prediction",
                "window_index": window_index,

                "true_artifact": true_class,
                "predicted_artifact": predicted_class,

                "model_probability": model_probability,
                "class_probabilities": (
                    prediction["class_probabilities"]
                ),

                "trust": trust.to_dict(),

                "beats": beats,
                "beat_count": len(beats),
                "window_start_sample": start_sample,

                "experimental": True,
                "imu_simulated": True,
                "ecg_replayed": True,
                "artifact_injected": True,
                "clinical_validated": False,
            })

            # ------------------------------------------------
            # 10. Send NEW audit event
            # ------------------------------------------------

            await websocket.send_json({
                "type": "artifact_audit",
                "window_index": window_index,
                "audit": audit,
            })

            # ------------------------------------------------
            # 11. Stream existing ECG waveform frames
            # ------------------------------------------------

            for offset in range(
                0,
                len(corrupted),
                FRAME_SIZE,
            ):
                end = min(
                    offset + FRAME_SIZE,
                    len(corrupted),
                )

                await websocket.send_json({
                    "type": "artifact_ecg_frame",
                    "window_index": window_index,
                    "sample_offset": offset,
                    "fs": fs,

                    "ecg_reference": (
                        reference[offset:end].tolist()
                    ),

                    "ecg_corrupted": (
                        corrupted[offset:end].tolist()
                    ),

                    "imu_simulated_values": (
                        imu[offset:end].tolist()
                    ),

                    "ecg_units": "uncalibrated_digital_values",
                    "ecg_replayed": True,
                    "artifact_injected": True,
                    "imu_simulated": True,
                })

                await asyncio.sleep(FRAME_DELAY)

        # ----------------------------------------------------
        # 12. End demonstration
        # ----------------------------------------------------

        await websocket.send_json({
            "type": "artifact_demo_complete",
            "windows_streamed": window_count,
        })

    except WebSocketDisconnect:
        print("TRACE artefact demo client disconnected.")

    except Exception as exc:
        print("TRACE artefact stream error:", exc)

        try:
            await websocket.send_json({
                "type": "error",
                "message": "Artefact demo failed.",
            })
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass
