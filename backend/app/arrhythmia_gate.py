
"""
TRACE — Experimental Arrhythmia Verification Gate

Routes independent model predictions according to
beat-level signal trust.

This module does not detect beats or classify ECG.
It does not use reference annotation symbols.

Research prototype. Not for clinical use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import math


TrustLevel = Literal["HIGH", "REVIEW", "UNRELIABLE"]

Decision = Literal["ACCEPT", "REVIEW", "ABSTAIN"]

POLICY_VERSION = "trace-arrhythmia-gate-v0.1"


@dataclass(frozen=True)
class VerificationResult:
    beat_sample: int
    trust_level: TrustLevel
    decision: Decision
    model_prediction: str | None
    model_probability: float | None
    released_prediction: str | None
    reason: str
    policy_version: str = POLICY_VERSION
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def verify_arrhythmia_prediction(
    beat_sample: int,
    trust_level: TrustLevel,
    model_prediction: str | None,
    model_probability: float | None,
) -> VerificationResult:
    """
    Apply a conservative trust gate to an independently
    generated arrhythmia model prediction.

    ACCEPT:
        HIGH trust with a valid model prediction.

    REVIEW:
        REVIEW trust with a valid model prediction.
        Prediction is withheld from automatic release.

    ABSTAIN:
        UNRELIABLE trust or unavailable prediction.

    No clinical correctness is implied by ACCEPT.
    """

    if beat_sample < 0:
        raise ValueError("Beat sample cannot be negative.")

    if trust_level not in (
        "HIGH",
        "REVIEW",
        "UNRELIABLE",
    ):
        raise ValueError("Invalid trust level.")

    if model_prediction is not None:
        if not isinstance(model_prediction, str):
            raise ValueError("Prediction must be a string.")

        if not model_prediction.strip():
            raise ValueError("Prediction cannot be empty.")

    if model_probability is not None:
        if not isinstance(
            model_probability, (int, float)
        ):
            raise ValueError("Invalid probability type.")

        if not math.isfinite(model_probability):
            raise ValueError("Probability must be finite.")

        if not 0.0 <= model_probability <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

    # A prediction and its probability must be supplied
    # together. This prevents partially populated results.
    if (model_prediction is None) != (
        model_probability is None
    ):
        raise ValueError(
            "Prediction and probability must both be "
            "present or both be absent."
        )

    if trust_level == "UNRELIABLE":
        decision: Decision = "ABSTAIN"
        released_prediction = None
        reason = (
            "Local ECG quality is unreliable. "
            "Automatic arrhythmia interpretation is withheld."
        )

    elif model_prediction is None:
        decision = "ABSTAIN"
        released_prediction = None
        reason = (
            "No independent model prediction is available."
        )

    elif trust_level == "REVIEW":
        decision = "REVIEW"
        released_prediction = None
        reason = (
            "Signal quality requires review. "
            "The experimental prediction is not "
            "automatically released."
        )

    else:
        decision = "ACCEPT"
        released_prediction = model_prediction
        reason = (
            "Signal quality passed the experimental "
            "trust gate. This does not establish "
            "clinical correctness."
        )

    return VerificationResult(
        beat_sample=beat_sample,
        trust_level=trust_level,
        decision=decision,
        model_prediction=model_prediction,
        model_probability=model_probability,
        released_prediction=released_prediction,
        reason=reason,
    )
