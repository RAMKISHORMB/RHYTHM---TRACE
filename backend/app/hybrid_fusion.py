
"""
TRACE — Hybrid Sensor Fusion
MEDAITHON'26 | PS-001

Combines:
1. Reference-free ECG quality measurements
2. Motion intelligence
3. Local dropout evidence
4. Optional external artefact-classifier prediction

This module routes signals for review. It does not
detect arrhythmias, make clinical diagnoses, or
suppress possible cardiac-event candidates.

Experimental research software only.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from app.ecg_quality import analyze_ecg_quality
from app.motion_intelligence import analyze_motion
from app.dropout_detection import detect_local_dropout


KNOWN_CLASSES = {
    "clean",
    "baseline_wander",
    "walking",
    "electrode_pop",
    "lead_dropout",
}


@dataclass
class HybridFusionResult:
    decision: str
    motion_level: str
    motion_rms: float
    ecg_quality_flags: list[str]
    possible_dropout: bool
    dropout_duration_ms: float
    artifact_prediction: Optional[str]
    artifact_confidence: Optional[float]
    classifier_available: bool
    evidence_conflict: bool
    reasons: list[str]
    cardiac_event_policy: str
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def analyze_hybrid_fusion(
    ecg,
    imu,
    fs=250,
    artifact_prediction=None,
    artifact_confidence=None,
    source="simulated_imu",
):
    """
    Generate a conservative sensor-fusion review decision.

    ECG and IMU must already be time-aligned.

    The optional classifier prediction must come from
    an independently invoked artefact classifier.
    It is never inferred from the dataset's true label.
    """

    ecg = np.asarray(ecg, dtype=np.float64)
    imu = np.asarray(imu, dtype=np.float64)

    if ecg.ndim != 1 or len(ecg) < 100:
        raise ValueError("Invalid ECG input.")

    if imu.ndim == 1:
        imu = imu[:, None]

    if imu.ndim != 2 or imu.shape[1] not in (1, 3):
        raise ValueError("IMU must have 1 or 3 channels.")

    if len(ecg) != len(imu):
        raise ValueError("ECG and IMU must be aligned.")

    if not np.all(np.isfinite(ecg)):
        raise ValueError("ECG contains invalid values.")

    if not np.all(np.isfinite(imu)):
        raise ValueError("IMU contains invalid values.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Invalid sensor source.")

    if artifact_prediction is not None:
        if not isinstance(artifact_prediction, str):
            raise ValueError("Prediction must be a class name.")

        if artifact_prediction not in KNOWN_CLASSES:
            raise ValueError("Unknown artefact class.")

    if artifact_confidence is not None:
        if artifact_prediction is None:
            raise ValueError(
                "Confidence requires an artefact prediction."
            )

        if (
            not np.isfinite(artifact_confidence)
            or not 0 <= artifact_confidence <= 1
        ):
            raise ValueError("Invalid classifier confidence.")

        artifact_confidence = float(artifact_confidence)

    quality = analyze_ecg_quality(ecg, fs=fs)
    motion = analyze_motion(imu, source=source)
    dropout = detect_local_dropout(ecg, fs=fs)

    reasons = []
    conflict = False

    # A dropout warning is independent of motion.
    if dropout.possible_dropout:
        reasons.append(
            "Sustained low-amplitude ECG segment "
            "requires signal-quality review."
        )

    if quality.quality_flags:
        reasons.append(
            "Provisional ECG quality flags: "
            + ", ".join(quality.quality_flags)
        )

    if motion.activity_level == "HIGH":
        reasons.append(
            "High simulated motion activity is present."
        )
    elif motion.activity_level == "MODERATE":
        reasons.append(
            "Moderate simulated motion activity is present."
        )

    classifier_available = artifact_prediction is not None

    if classifier_available:
        reasons.append(
            "Artefact classifier predicts: "
            + artifact_prediction
        )

        if artifact_confidence is not None:
            reasons.append(
                "Reported classifier confidence: "
                f"{artifact_confidence:.3f}; "
                "not calibrated for clinical use."
            )

        # Disagreement is retained as review evidence.
        if artifact_prediction == "clean":
            if (
                dropout.possible_dropout
                or motion.activity_level == "HIGH"
                or bool(quality.quality_flags)
            ):
                conflict = True

        if artifact_prediction == "lead_dropout":
            if not dropout.possible_dropout:
                conflict = True

        if artifact_prediction == "walking":
            if motion.activity_level == "LOW":
                conflict = True

        if conflict:
            reasons.append(
                "Classifier and independent sensor "
                "evidence disagree; review required."
            )
    else:
        reasons.append(
            "Artefact classifier not supplied; "
            "decision uses heuristic evidence only."
        )

    # These are engineering review-routing states,
    # not clinically validated trust classifications.
    if conflict:
        decision = "REVIEW_CONFLICT"

    elif dropout.possible_dropout:
        decision = "REVIEW_DROPOUT"

    elif (
        classifier_available
        and artifact_prediction != "clean"
    ):
        decision = "REVIEW_ARTEFACT"

    elif motion.activity_level == "HIGH":
        decision = "REVIEW_MOTION"

    elif quality.quality_flags:
        decision = "REVIEW_ECG_QUALITY"

    elif motion.activity_level == "MODERATE":
        decision = "REVIEW_MOTION"

    else:
        decision = "NO_WARNING_DETECTED"
        reasons.append(
            "No warning triggered by the available "
            "experimental evidence. ECG reliability "
            "is not established."
        )

    return HybridFusionResult(
        decision=decision,
        motion_level=motion.activity_level,
        motion_rms=motion.rms,
        ecg_quality_flags=list(quality.quality_flags),
        possible_dropout=dropout.possible_dropout,
        dropout_duration_ms=dropout.longest_low_amplitude_ms,
        artifact_prediction=artifact_prediction,
        artifact_confidence=artifact_confidence,
        classifier_available=classifier_available,
        evidence_conflict=conflict,
        reasons=reasons,
        cardiac_event_policy=(
            "Never automatically dismiss or suppress "
            "a cardiac-event candidate solely because "
            "motion or artefact evidence is present."
        ),
    )
