
"""
TRACE — Stage 12.1
Unified ECG Analysis Pipeline

Integrates:
- Virtual wearable ECG generation
- ECG signal-quality assessment
- Existing artefact inference
- Conservative motion-compensation gate
- Explainable review routing

Research prototype only.

Important limitations:
- MIT-BIH ECG replay, not live wearable acquisition.
- Simulated 1D motion proxy, not a physical IMU.
- Synthetic artefact classifier is not externally validated.
- Four-second windows are not used for independent
  transient-event diagnosis.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.virtual_wearable import generate_wearable_window
from app.artifact_inference import ArtifactInference
from app.ecg_quality import analyze_ecg_quality
from app.safe_motion_compensation import safe_motion_compensate
from app.event_trust_router import route_event


FS = 250

SCENARIOS = {
    "REST",
    "WALKING",
    "ELECTRODE_POP",
}


@dataclass
class UnifiedWindowResult:
    scenario: str
    start_seconds: float
    sample_rate_hz: int
    sample_count: int
    duration_seconds: float
    artifact_label: str
    quality_flags: list[str]
    compensation_decision: str
    compensation_reason: str
    compensation_accepted: bool
    trust_status: str
    trust_reasons: list[str]
    observed_ecg: list[float]
    output_ecg: list[float]
    simulated_motion: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "start_seconds": self.start_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "duration_seconds": self.duration_seconds,
            "artifact_label": self.artifact_label,
            "quality_flags": self.quality_flags,
            "compensation": {
                "decision": self.compensation_decision,
                "reason": self.compensation_reason,
                "accepted": self.compensation_accepted,
            },
            "trust_status": self.trust_status,
            "trust_reasons": self.trust_reasons,
            "observed_ecg": self.observed_ecg,
            "output_ecg": self.output_ecg,
            "simulated_motion": self.simulated_motion,
            "metadata": self.metadata,
        }


def _extract_artifact_label(prediction: dict) -> str:
    """
    Handle common dictionary outputs from the
    existing artefact inference component.

    If the schema differs, return 'unknown'
    rather than inventing a clean classification.
    """

    if not isinstance(prediction, dict):
        return "unknown"

    for key in (
        "predicted_class",
        "predicted_label",
        "artifact_label",
        "label",
        "class_name",
    ):
        value = prediction.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return "unknown"


def analyze_unified_window(
    scenario: str = "REST",
    start_seconds: float = 180.0,
    seed: int = 42,
) -> UnifiedWindowResult:
    """
    Analyze one simulated four-second ECG window.

    Does not detect new cardiac events.
    """

    scenario = scenario.upper().strip()

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    wearable = generate_wearable_window(
        scenario=scenario,
        start_seconds=start_seconds,
        seed=seed,
    )

    observed = np.asarray(
        wearable.ecg_observed,
        dtype=np.float64,
    )

    motion = np.asarray(
        wearable.imu_simulated,
        dtype=np.float64,
    )

    if observed.ndim != 1 or motion.ndim != 1:
        raise ValueError("Expected 1D ECG and motion arrays.")

    if len(observed) != len(motion):
        raise ValueError("ECG and motion lengths differ.")

    if len(observed) != 4 * FS:
        raise ValueError(
            "Expected a four-second window at 250 Hz."
        )

    if not np.isfinite(observed).all():
        raise ValueError("Observed ECG is not finite.")

    if not np.isfinite(motion).all():
        raise ValueError("Simulated motion is not finite.")

    quality = analyze_ecg_quality(
        observed,
        fs=FS,
    )

    flags = list(quality.quality_flags)

    classifier = ArtifactInference()

    prediction = classifier.predict(
        observed,
        motion,
    )

    artifact_label = _extract_artifact_label(
        prediction
    )

    compensation = safe_motion_compensate(
        observed_ecg=observed,
        motion_reference=motion,
        fs=FS,
    )

    output = np.asarray(
        compensation["output_ecg"],
        dtype=np.float64,
    )

    # Scenario knowledge is used only as
    # simulated motion evidence. It is NOT
    # a claim that a real IMU detected motion.
    simulated_motion_suspected = (
        scenario != "REST"
    )

    # A routing placeholder represents a
    # window review item, not a cardiac event.
    routing = route_event(
        event={
            "item_type": "ECG_WINDOW_REVIEW",
            "scenario": scenario,
            "start_seconds": float(start_seconds),
        },
        artifact_label=artifact_label,
        quality_flags=flags,
        compensation=compensation,
        motion_suspected=simulated_motion_suspected,
    )

    return UnifiedWindowResult(
        scenario=scenario,
        start_seconds=float(start_seconds),
        sample_rate_hz=FS,
        sample_count=len(observed),
        duration_seconds=len(observed) / FS,
        artifact_label=artifact_label,
        quality_flags=flags,
        compensation_decision=str(
            compensation["decision"]
        ),
        compensation_reason=str(
            compensation["reason"]
        ),
        compensation_accepted=bool(
            compensation["accepted"]
        ),
        trust_status=str(
            routing["trust_status"]
        ),
        trust_reasons=list(
            routing["trust_reasons"]
        ),
        observed_ecg=observed.tolist(),
        output_ecg=output.tolist(),
        simulated_motion=motion.tolist(),
        metadata={
            "source": "MIT_BIH_REPLAY_WITH_SYNTHETIC_MOTION",
            "motion_source": "SIMULATED_1D_PROXY",
            "artifact_model": "SYNTHETIC_LIGHTGBM",
            "processing_mode": "OFFLINE",
            "cardiac_event_detected": False,
            "event_detection_performed": False,
            "requires_review": True,
            "clinical_validated": False,
            "artifact_prediction": {
                "raw_keys": (
                    list(prediction.keys())
                    if isinstance(prediction, dict)
                    else []
                ),
                **(prediction if isinstance(prediction, dict) else {}),
            },
        },
    )
