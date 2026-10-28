
"""
TRACE — Unified Virtual Wearable Generator
MEDAITHON'26 | Cardiology PS-001

Combines recorded MIT-BIH ECG with a simulated
single-channel motion proxy and optional injected
motion artefacts.

Research simulation only.
Not real IMU hardware or live patient monitoring.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ECG_PATH = (
    ROOT / "data" / "processed" / "100_250hz.npz"
)

FS = 250
WINDOW_SECONDS = 4
WINDOW_SAMPLES = FS * WINDOW_SECONDS

SCENARIOS = {
    "REST",
    "WALKING",
    "ELECTRODE_POP",
}


@dataclass
class WearableWindow:
    scenario: str
    recording_id: str
    fs: int
    start_seconds: float
    end_seconds: float
    ecg_reference: np.ndarray
    ecg_observed: np.ndarray
    imu_simulated: np.ndarray
    motion_source: str = "synthetic_single_channel_proxy"

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "recording_id": self.recording_id,
            "fs": self.fs,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "ecg_reference": self.ecg_reference.tolist(),
            "ecg_observed": self.ecg_observed.tolist(),
            "imu_simulated": self.imu_simulated.tolist(),
            "motion_source": self.motion_source,
            "research_only": True,
            "clinical_validated": False,
        }


def load_recorded_ecg() -> np.ndarray:
    if not ECG_PATH.exists():
        raise FileNotFoundError(
            f"Recorded ECG not found: {ECG_PATH}"
        )

    with np.load(
        ECG_PATH,
        allow_pickle=False,
    ) as data:
        ecg = np.asarray(
            data["ecg"],
            dtype=np.float64,
        )
        fs = int(data["fs"])

    if fs != FS:
        raise ValueError(
            f"Expected {FS} Hz, received {fs} Hz."
        )

    if ecg.ndim != 2 or ecg.shape[1] < 1:
        raise ValueError(
            f"Unexpected ECG shape: {ecg.shape}"
        )

    signal = ecg[:, 0]

    if not np.isfinite(signal).all():
        raise ValueError(
            "Recorded ECG contains invalid values."
        )

    return signal


def generate_wearable_window(
    scenario: str = "REST",
    start_seconds: float = 180.0,
    seed: int = 42,
) -> WearableWindow:
    scenario = scenario.strip().upper()

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    if not np.isfinite(start_seconds):
        raise ValueError(
            "Start time must be finite."
        )

    if start_seconds < 0:
        raise ValueError(
            "Start time cannot be negative."
        )

    signal = load_recorded_ecg()

    start_sample = round(start_seconds * FS)
    end_sample = start_sample + WINDOW_SAMPLES

    if end_sample > len(signal):
        raise ValueError(
            "Requested window exceeds recording."
        )

    reference = signal[
        start_sample:end_sample
    ].copy()

    observed = reference.copy()

    rng = np.random.default_rng(seed)

    t = np.arange(
        WINDOW_SAMPLES,
        dtype=np.float64,
    ) / FS

    if scenario == "REST":
        motion = (
            0.015 * np.sin(2 * np.pi * 0.35 * t)
            + rng.normal(
                0,
                0.005,
                WINDOW_SAMPLES,
            )
        )

    elif scenario == "WALKING":
        motion = (
            0.65 * np.sin(2 * np.pi * 1.8 * t)
            + 0.18 * np.sin(2 * np.pi * 3.6 * t)
            + rng.normal(
                0,
                0.06,
                WINDOW_SAMPLES,
            )
        )

        observed += (
            0.18 * np.sin(2 * np.pi * 1.8 * t)
            + 0.08 * np.sin(2 * np.pi * 0.4 * t)
            + rng.normal(
                0,
                0.035,
                WINDOW_SAMPLES,
            )
        )

    else:
        motion = (
            0.03 * np.sin(2 * np.pi * 0.6 * t)
            + rng.normal(
                0,
                0.01,
                WINDOW_SAMPLES,
            )
        )

        # Synthetic electrode transient centered
        # within the four-second ECG segment.
        center = 2.0
        width = 0.035

        pop = 1.5 * np.exp(
            -0.5 * ((t - center) / width) ** 2
        )

        observed += pop

    if not (
        np.isfinite(observed).all()
        and np.isfinite(motion).all()
    ):
        raise ValueError(
            "Generated signals contain invalid values."
        )

    return WearableWindow(
        scenario=scenario,
        recording_id="MITBIH-100",
        fs=FS,
        start_seconds=start_sample / FS,
        end_seconds=end_sample / FS,
        ecg_reference=reference,
        ecg_observed=observed,
        imu_simulated=motion,
    )
