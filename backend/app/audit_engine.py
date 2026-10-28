
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


TrustLevel = Literal["HIGH", "REVIEW", "UNRELIABLE"]
AuditAction = Literal[
    "PRESERVED",
    "CANDIDATE_FILTERED",
    "ABSTAINED",
]


@dataclass(frozen=True)
class AuditEntry:
    event_id: str
    timestamp_utc: str
    window_index: int
    artifact_class: str
    trust_level: TrustLevel
    action: AuditAction
    explanation: str
    experimental: bool
    clinical_validated: bool


def create_audit_entry(
    window_index: int,
    artifact_class: str,
    trust_level: TrustLevel,
    action: AuditAction,
) -> dict:
    if window_index < 0:
        raise ValueError("Window index cannot be negative.")

    if not artifact_class.strip():
        raise ValueError("Artifact class cannot be empty.")

    if trust_level not in (
        "HIGH",
        "REVIEW",
        "UNRELIABLE",
    ):
        raise ValueError("Invalid trust level.")

    if action not in (
        "PRESERVED",
        "CANDIDATE_FILTERED",
        "ABSTAINED",
    ):
        raise ValueError("Invalid audit action.")

    if trust_level == "UNRELIABLE":
        if action != "ABSTAINED":
            raise ValueError(
                "UNRELIABLE windows must abstain."
            )

        explanation = (
            "Signal quality is unreliable. "
            "No reconstructed waveform is authorized."
        )

    elif trust_level == "HIGH":
        if action != "PRESERVED":
            raise ValueError(
                "HIGH-trust windows must preserve raw ECG."
            )

        explanation = (
            "Signal trust is high. "
            "The original recorded ECG is preserved."
        )

    elif (
        trust_level == "REVIEW"
        and artifact_class == "baseline_wander"
        and action == "CANDIDATE_FILTERED"
    ):
        explanation = (
            "Baseline wander was identified. "
            "An experimental cleaning candidate "
            "was generated for review only."
        )

    elif trust_level == "REVIEW" and action == "PRESERVED":
        explanation = (
            "Signal requires review. "
            "No approved cleaning action was applied; "
            "the original ECG was preserved."
        )

    elif trust_level == "REVIEW" and action == "ABSTAINED":
        explanation = (
            "Signal requires review, but the cleaning "
            "policy abstained from reconstruction."
        )

    else:
        raise ValueError(
            "Action is inconsistent with the trust policy."
        )

    entry = AuditEntry(
        event_id=str(uuid4()),
        timestamp_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        window_index=window_index,
        artifact_class=artifact_class,
        trust_level=trust_level,
        action=action,
        explanation=explanation,
        experimental=True,
        clinical_validated=False,
    )

    return asdict(entry)
