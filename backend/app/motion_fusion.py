
"""
TRACE — ECG/Motion Evidence Fusion
PS-001 experimental research prototype.

Uses aligned ECG reference/corrupted pairs and a
simulated motion proxy to measure temporal association.

This is NOT clinical event detection or proof of
motion-caused artefacts.
"""

from dataclasses import dataclass, asdict
import numpy as np

from app.motion_intelligence import analyze_motion


@dataclass
class FusionResult:
    motion_rms: float
    motion_level: str
    ecg_difference_rms: float
    temporal_correlation: float | None
    evidence_category: str
    explanation: str
    reference_available: bool = True
    simulated_motion: bool = True
    experimental: bool = True

    def to_dict(self):
        return asdict(self)


def analyze_ecg_motion(
    ecg_reference,
    ecg_corrupted,
    imu,
    motion_threshold=0.05,
    correlation_threshold=0.35,
):
    reference = np.asarray(
        ecg_reference, dtype=np.float64
    )
    corrupted = np.asarray(
        ecg_corrupted, dtype=np.float64
    )
    motion = np.asarray(imu, dtype=np.float64)

    if reference.ndim != 1 or corrupted.ndim != 1:
        raise ValueError("ECG arrays must be 1D.")

    if len(reference) < 20:
        raise ValueError("ECG window is too short.")

    if reference.shape != corrupted.shape:
        raise ValueError("ECG arrays must be aligned.")

    if motion.ndim == 2 and motion.shape[1] in (1, 3):
        pass
    elif motion.ndim != 1:
        raise ValueError("Invalid motion shape.")

    if len(motion) != len(reference):
        raise ValueError(
            "Motion must already be aligned to ECG."
        )

    if not (
        np.all(np.isfinite(reference))
        and np.all(np.isfinite(corrupted))
        and np.all(np.isfinite(motion))
    ):
        raise ValueError("Non-finite sensor values.")

    if not 0 <= motion_threshold:
        raise ValueError("Invalid motion threshold.")

    if not 0 <= correlation_threshold <= 1:
        raise ValueError("Invalid correlation threshold.")

    motion_result = analyze_motion(motion)

    # Difference is available only in controlled
    # experiments with a reference ECG.
    difference = corrupted - reference
    difference_rms = float(
        np.sqrt(np.mean(difference ** 2))
    )

    # Motion envelope: remove constant channel offsets.
    if motion.ndim == 1:
        motion_matrix = motion[:, None]
    else:
        motion_matrix = motion

    centered_motion = (
        motion_matrix - np.mean(motion_matrix, axis=0)
    )

    motion_envelope = np.linalg.norm(
        centered_motion, axis=1
    )

    # Difference magnitude is a controlled-experiment
    # artefact proxy, not a clinical event score.
    difference_envelope = np.abs(difference)

    # Smooth both envelopes to emphasize broad
    # temporal changes rather than ECG morphology.
    kernel = np.ones(25, dtype=np.float64) / 25

    smooth_motion = np.convolve(
        motion_envelope, kernel, mode="same"
    )
    smooth_difference = np.convolve(
        difference_envelope, kernel, mode="same"
    )

    if (
        np.std(smooth_motion) < 1e-10
        or np.std(smooth_difference) < 1e-10
    ):
        correlation = None
    else:
        correlation = float(np.corrcoef(
            smooth_motion, smooth_difference
        )[0, 1])

    if motion_result.rms < motion_threshold:
        category = "LOW_MOTION_EVIDENCE"
        explanation = (
            "Motion activity is low under the "
            "experimental threshold. This does not "
            "guarantee ECG reliability."
        )
    elif (
        correlation is not None
        and correlation >= correlation_threshold
    ):
        category = "TEMPORALLY_ASSOCIATED"
        explanation = (
            "Motion and ECG difference envelopes "
            "show positive temporal association. "
            "This does not establish causation or "
            "exclude a genuine cardiac event."
        )
    else:
        category = "MOTION_PRESENT_UNCONFIRMED"
        explanation = (
            "Motion is present, but strong positive "
            "temporal association with the ECG "
            "difference was not demonstrated."
        )

    return FusionResult(
        motion_rms=motion_result.rms,
        motion_level=motion_result.activity_level,
        ecg_difference_rms=difference_rms,
        temporal_correlation=correlation,
        evidence_category=category,
        explanation=explanation,
    )
