
"""
TRACE — Virtual Wearable Analysis
MEDAITHON'26 | Cardiology PS-001
Stage 11.2

Integrates:
- Recorded MIT-BIH ECG
- Simulated single-channel motion proxy
- LightGBM artefact classification
- Experimental ECG quality assessment

Research prototype only.
Not clinically validated.
"""

from functools import lru_cache

import numpy as np

from app.virtual_wearable import (
    generate_wearable_window,
)
from app.artifact_inference import ArtifactInference
from app.ecg_quality import analyze_ecg_quality


# ============================================================
# CLASSIFIER INITIALIZATION
# ============================================================

@lru_cache(maxsize=1)
def get_classifier() -> ArtifactInference:
    """
    Load the existing trained classifier once.

    Avoids repeatedly loading the model for
    every virtual wearable analysis.
    """
    return ArtifactInference()


# ============================================================
# VIRTUAL WEARABLE ANALYSIS
# ============================================================

def analyze_virtual_wearable(
    scenario: str = "REST",
    start_seconds: float = 180.0,
    seed: int = 42,
) -> dict:
    """
    Analyze one four-second virtual wearable window.

    Returns:
        Scenario metadata
        Motion RMS
        Reference corruption RMS
        Artefact classifier prediction
        ECG quality flags

    Note:
        Reference corruption RMS is available only
        because this is an offline simulation.

        It must not be treated as a measurement
        available from a real wearable.
    """

    # --------------------------------------------------------
    # 1. GENERATE SYNCHRONIZED DATA
    # --------------------------------------------------------

    window = generate_wearable_window(
        scenario=scenario,
        start_seconds=start_seconds,
        seed=seed,
    )

    reference_ecg = np.asarray(
        window.ecg_reference,
        dtype=np.float64,
    )

    observed_ecg = np.asarray(
        window.ecg_observed,
        dtype=np.float64,
    )

    simulated_motion = np.asarray(
        window.imu_simulated,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 2. VALIDATE INPUTS
    # --------------------------------------------------------

    sample_count = len(observed_ecg)

    if sample_count == 0:
        raise ValueError(
            "Virtual wearable ECG window is empty."
        )

    if not (
        len(reference_ecg)
        == len(observed_ecg)
        == len(simulated_motion)
    ):
        raise ValueError(
            "ECG and motion sample lengths differ."
        )

    if not all(
        np.isfinite(signal).all()
        for signal in (
            reference_ecg,
            observed_ecg,
            simulated_motion,
        )
    ):
        raise ValueError(
            "Virtual wearable contains invalid samples."
        )

    # --------------------------------------------------------
    # 3. ARTEFACT CLASSIFICATION
    # --------------------------------------------------------

    prediction = get_classifier().predict(
        observed_ecg,
        simulated_motion,
    )

    # The existing TRACE classifier returns a dict.
    if not isinstance(prediction, dict):
        raise TypeError(
            "Expected artefact prediction dictionary."
        )

    # --------------------------------------------------------
    # 4. ECG QUALITY ANALYSIS
    # --------------------------------------------------------

    quality = analyze_ecg_quality(
        observed_ecg,
        fs=window.fs,
    )

    quality_flags = list(
        quality.quality_flags
    )

    # --------------------------------------------------------
    # 5. SIMULATED MOTION INTENSITY
    # --------------------------------------------------------

    motion_rms = float(
        np.sqrt(
            np.mean(simulated_motion ** 2)
        )
    )

    # --------------------------------------------------------
    # 6. REFERENCE CORRUPTION MEASUREMENT
    # --------------------------------------------------------

    corruption_rms = float(
        np.sqrt(
            np.mean(
                (
                    observed_ecg - reference_ecg
                ) ** 2
            )
        )
    )

    # This measurement is for simulation evaluation.
    # A real wearable cannot directly access
    # an uncorrupted reference ECG.

    # --------------------------------------------------------
    # 7. RETURN UNIFIED ANALYSIS
    # --------------------------------------------------------

    return {
        "scenario": window.scenario,
        "recording_id": window.recording_id,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
        "sampling_rate": window.fs,
        "sample_count": sample_count,

        "motion_source": window.motion_source,
        "motion_available": True,
        "motion_is_simulated": True,
        "motion_rms": motion_rms,
        "reference_ecg": reference_ecg.tolist(),
        "observed_ecg": observed_ecg.tolist(),
        "simulated_motion": simulated_motion.tolist(),

        "reference_corruption_rms": corruption_rms,
        "reference_available_only_in_simulation": True,

        "artifact_prediction": prediction,

        "quality_flags": quality_flags,
        "quality_method": (
            "experimental_ecg_quality_heuristics"
        ),

        "research_only": True,
        "clinical_validated": False,
        "live_patient_monitoring": False,
    }
