
"""
TRACE — Reference-Free Wearable Sensor Fusion
MEDAITHON PS-001

Uses observed ECG and aligned motion measurements.

Research prototype:
- No clean reference ECG required
- No cardiac event detection
- No clinical interpretation
- Thresholds are provisional and unvalidated
"""

from dataclasses import dataclass, asdict

import numpy as np

from app.motion_intelligence import analyze_motion


@dataclass
class WearableFusionResult:
    ecg_rms: float
    ecg_variability: float
    ecg_peak_to_peak: float
    flatline_fraction: float
    motion_rms: float
    motion_level: str
    motion_ecg_correlation: float | None
    evidence_state: str
    explanation: str
    source: str
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def _moving_average(signal, width):
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(signal, kernel, mode="same")


def analyze_wearable_fusion(
    ecg,
    imu,
    fs=250,
    source="simulated_imu",
):
    ecg = np.asarray(ecg, dtype=np.float64)
    imu = np.asarray(imu, dtype=np.float64)

    if ecg.ndim != 1 or len(ecg) < 100:
        raise ValueError(
            "ECG must be a 1D array of at least 100 samples."
        )

    if imu.ndim == 1:
        imu = imu[:, None]

    if imu.ndim != 2 or imu.shape[1] not in (1, 3):
        raise ValueError("IMU must have 1 or 3 channels.")

    if len(imu) != len(ecg):
        raise ValueError("ECG and IMU must be aligned.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if not np.all(np.isfinite(ecg)):
        raise ValueError("ECG contains invalid values.")

    if not np.all(np.isfinite(imu)):
        raise ValueError("IMU contains invalid values.")

    motion = analyze_motion(imu, source=source)

    ecg_centered = ecg - np.mean(ecg)

    ecg_rms = float(
        np.sqrt(np.mean(ecg_centered ** 2))
    )
    ecg_variability = float(np.std(ecg))
    ecg_peak_to_peak = float(np.ptp(ecg))

    # Detect locally near-constant ECG samples.
    # The threshold is relative to the window range,
    # so it is not a calibrated dropout detector.
    local_difference = np.abs(np.diff(ecg))
    flatline_threshold = max(
        1e-9,
        0.0005 * ecg_peak_to_peak,
    )

    flatline_fraction = float(np.mean(
        local_difference < flatline_threshold
    ))

    # Motion magnitude after removal of each
    # channel's constant offset.
    centered_imu = imu - np.mean(
        imu, axis=0, keepdims=True
    )

    motion_magnitude = np.linalg.norm(
        centered_imu, axis=1
    )

    # Observed ECG change magnitude is a
    # reference-free activity proxy.
    # It is NOT an artefact ground-truth signal.
    ecg_change = np.abs(
        np.diff(ecg, prepend=ecg[0])
    )

    width = max(3, int(round(fs * 0.1)))
    width = min(width, len(ecg))

    smooth_motion = _moving_average(
        motion_magnitude, width
    )
    smooth_ecg_change = _moving_average(
        ecg_change, width
    )

    if (
        np.std(smooth_motion) < 1e-10
        or np.std(smooth_ecg_change) < 1e-10
    ):
        correlation = None
    else:
        correlation = float(np.corrcoef(
            smooth_motion,
            smooth_ecg_change,
        )[0, 1])

    # These states describe preliminary evidence,
    # not verified clinical signal reliability.
    if flatline_fraction >= 0.8:
        state = "POSSIBLE_DROPOUT"
        explanation = (
            "A large fraction of adjacent ECG samples "
            "show very small changes. Review for possible "
            "dropout or a low-variation recording."
        )

    elif motion.activity_level == "HIGH":
        state = "MOTION_REVIEW"
        explanation = (
            "High motion activity is present. ECG "
            "interpretation may require review. "
            "Motion alone does not prove artefact "
            "or exclude a cardiac event."
        )

    elif (
        correlation is not None
        and correlation >= 0.35
        and motion.activity_level == "MODERATE"
    ):
        state = "MOTION_ASSOCIATED_REVIEW"
        explanation = (
            "Observed ECG changes and motion show "
            "positive temporal association. "
            "This is not proof of motion-caused artefact."
        )

    else:
        state = "NO_STRONG_MOTION_WARNING"
        explanation = (
            "No strong motion warning was triggered "
            "by these provisional heuristics. "
            "This does not establish ECG reliability."
        )

    return WearableFusionResult(
        ecg_rms=ecg_rms,
        ecg_variability=ecg_variability,
        ecg_peak_to_peak=ecg_peak_to_peak,
        flatline_fraction=flatline_fraction,
        motion_rms=motion.rms,
        motion_level=motion.activity_level,
        motion_ecg_correlation=correlation,
        evidence_state=state,
        explanation=explanation,
        source=source,
    )
