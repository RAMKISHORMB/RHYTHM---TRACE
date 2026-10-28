
"""
TRACE experimental signal trust engine.

This is a transparent, heuristic research policy.
It is not a calibrated signal-quality index and
must not be used for clinical decision-making.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


TrustLevel = Literal["HIGH", "REVIEW", "UNRELIABLE"]

CLASS_TRUST = {
    "clean": "HIGH",
    "baseline_wander": "REVIEW",
    "walking": "REVIEW",
    "electrode_pop": "UNRELIABLE",
    "lead_dropout": "UNRELIABLE",
}

# Engineering thresholds for the prototype.
# These require validation before real-world use.
MIN_MODEL_PROBABILITY = 0.70
MAX_FLATLINE_FRACTION = 0.20
FLATLINE_TOLERANCE = 1e-6


@dataclass
class TrustResult:
    level: TrustLevel
    score: float
    predicted_class: str
    model_probability: float
    flatline_fraction: float
    reasons: list[str]
    policy_version: str = "trace-trust-v0.1"
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_flatline_fraction(
    ecg: np.ndarray,
) -> float:
    """
    Estimate the fraction of adjacent sample pairs
    that are nearly identical.

    This is a simple dropout indicator, not a
    validated ECG signal-quality measurement.
    """
    signal = np.asarray(ecg, dtype=np.float64)

    if signal.ndim != 1 or signal.size < 2:
        raise ValueError(
            "ECG must be a one-dimensional array "
            "containing at least two samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite values.")

    differences = np.abs(np.diff(signal))

    return float(
        np.mean(differences <= FLATLINE_TOLERANCE)
    )


def evaluate_trust(
    ecg_window: np.ndarray,
    predicted_class: str,
    model_probability: float,
) -> TrustResult:
    """
    Assign an experimental signal trust level.

    The model probability is used only as an
    uncertainty flag. It is not treated as a
    calibrated probability of signal safety.
    """

    if predicted_class not in CLASS_TRUST:
        raise ValueError(
            f"Unknown artefact class: {predicted_class}"
        )

    probability = float(model_probability)

    if not np.isfinite(probability):
        raise ValueError(
            "Model probability must be finite."
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Model probability must be between 0 and 1."
        )

    flatline_fraction = calculate_flatline_fraction(
        ecg_window
    )

    reasons: list[str] = []

    # Initial rule-based trust classification.
    level: TrustLevel = CLASS_TRUST[predicted_class]

    if predicted_class == "clean":
        reasons.append(
            "Classifier predicts no injected artefact."
        )
    elif predicted_class in (
        "baseline_wander",
        "walking",
    ):
        reasons.append(
            "Classifier predicts an artefact that "
            "requires additional signal review."
        )
    else:
        reasons.append(
            "Classifier predicts a severe signal artefact."
        )

    # Low classifier certainty never upgrades trust.
    if probability < MIN_MODEL_PROBABILITY:
        reasons.append(
            "Classifier probability is below the "
            "experimental uncertainty threshold."
        )

        if level == "HIGH":
            level = "REVIEW"

    # Strong flatline evidence overrides classification.
    if flatline_fraction >= MAX_FLATLINE_FRACTION:
        level = "UNRELIABLE"
        reasons.append(
            "A high fraction of adjacent ECG samples "
            "are nearly identical, suggesting dropout."
        )

    # A descriptive policy score, NOT a probability.
    score_by_level = {
        "HIGH": 1.0,
        "REVIEW": 0.5,
        "UNRELIABLE": 0.0,
    }

    return TrustResult(
        level=level,
        score=score_by_level[level],
        predicted_class=predicted_class,
        model_probability=probability,
        flatline_fraction=flatline_fraction,
        reasons=reasons,
    )
