
"""
TRACE — LightGBM + Hybrid Sensor Fusion Pipeline
MEDAITHON'26 | PS-001

Combines the existing trained artefact classifier with
reference-free ECG quality and motion evidence.

Experimental research prototype only.
No cardiac diagnosis or autonomous alert suppression.
"""

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from app.artifact_inference import ArtifactInference
from app.hybrid_fusion import analyze_hybrid_fusion


@dataclass
class HybridPipelineResult:
    artifact: dict[str, Any]
    fusion: dict[str, Any]
    model_disagreement: bool
    requires_review: bool
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


class HybridPipeline:
    def __init__(self):
        # Load the trained model once.
        self.classifier = ArtifactInference()

    def analyze(
        self,
        ecg_window,
        imu_window,
    ) -> HybridPipelineResult:

        ecg = np.asarray(
            ecg_window,
            dtype=np.float64,
        )

        imu = np.asarray(
            imu_window,
            dtype=np.float64,
        )

        expected = (
            self.classifier.fs
            * self.classifier.window_seconds
        )

        if ecg.shape != (expected,):
            raise ValueError(
                f"Expected ECG shape ({expected},), "
                f"received {ecg.shape}"
            )

        if imu.shape != (expected,):
            raise ValueError(
                f"Expected IMU shape ({expected},), "
                f"received {imu.shape}"
            )

        if not np.all(np.isfinite(ecg)):
            raise ValueError(
                "ECG contains invalid values."
            )

        if not np.all(np.isfinite(imu)):
            raise ValueError(
                "IMU contains invalid values."
            )

        # 1. Actual LightGBM inference.
        artifact = self.classifier.predict(
            ecg,
            imu,
        )

        # 2. Reference-free hybrid evidence.
        fusion_result = analyze_hybrid_fusion(
            ecg=ecg,
            imu=imu,
            fs=self.classifier.fs,
            artifact_prediction=artifact["predicted_class"],
            artifact_confidence=artifact["model_probability"],
            source="simulated_imu",
        )

        fusion = fusion_result.to_dict()

        disagreement = bool(
            fusion_result.evidence_conflict
        )

        # Conservative review routing.
        # No warning is NOT clinical clearance.
        requires_review = bool(
            fusion_result.decision
            != "NO_WARNING_DETECTED"
        )

        return HybridPipelineResult(
            artifact=artifact,
            fusion=fusion,
            model_disagreement=disagreement,
            requires_review=requires_review,
        )
