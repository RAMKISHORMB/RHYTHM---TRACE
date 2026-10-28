
"""
TRACE — Experimental Beat-Level Signal Trust

Uses supplied reference beat positions, local ECG
measurements, and an upstream window trust level.

No beat detection or clinical diagnosis is performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np


TrustLevel = Literal["HIGH", "REVIEW", "UNRELIABLE"]

TRUST_RANK = {
    "UNRELIABLE": 0,
    "REVIEW": 1,
    "HIGH": 2,
}

POLICY_VERSION = "trace-beat-trust-v0.1"


@dataclass
class BeatTrustResult:
    sample: int
    symbol: str
    level: TrustLevel
    score: float
    flatline_fraction: float
    local_range: float
    reasons: list[str]
    annotation_source: str = "dataset_annotation"
    policy_version: str = POLICY_VERSION
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _lower_trust(
    current: TrustLevel,
    proposed: TrustLevel,
) -> TrustLevel:
    if TRUST_RANK[proposed] < TRUST_RANK[current]:
        return proposed

    return current


def evaluate_beat_trust(
    ecg_window: Sequence[float],
    beat_samples: Sequence[int],
    beat_symbols: Sequence[str],
    fs: int,
    window_trust: TrustLevel,
    local_radius_seconds: float = 0.20,
) -> list[BeatTrustResult]:
    """
    Evaluate local ECG quality around supplied beat positions.

    beat_samples are zero-based indices relative to ecg_window.

    Thresholds are engineering heuristics and have not
    been validated on clinical ECG datasets.
    """

    signal = np.asarray(ecg_window, dtype=np.float64)

    if signal.ndim != 1 or signal.size < 2:
        raise ValueError(
            "ECG must contain at least two samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains non-finite values.")

    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    if window_trust not in TRUST_RANK:
        raise ValueError("Invalid window trust level.")

    if len(beat_samples) != len(beat_symbols):
        raise ValueError(
            "Beat samples and symbols must have equal lengths."
        )

    if not 0 < local_radius_seconds <= 1.0:
        raise ValueError("Invalid local radius.")

    radius = max(
        1,
        int(round(local_radius_seconds * fs)),
    )

    results: list[BeatTrustResult] = []

    for sample, symbol in zip(
        beat_samples,
        beat_symbols,
    ):
        sample = int(sample)

        if not 0 <= sample < len(signal):
            raise ValueError(
                f"Beat sample {sample} is outside ECG window."
            )

        start = max(0, sample - radius)
        end = min(len(signal), sample + radius + 1)

        local = signal[start:end]

        differences = np.abs(np.diff(local))

        flatline_fraction = float(
            np.mean(differences <= 1e-6)
        )

        local_range = float(
            np.ptp(local)
        )

        level: TrustLevel = window_trust
        reasons = [
            f"Inherited window trust: {window_trust}."
        ]

        # Conservative local quality checks.
        if flatline_fraction >= 0.80:
            level = _lower_trust(
                level,
                "UNRELIABLE",
            )
            reasons.append(
                "Local region contains predominantly "
                "unchanged adjacent samples."
            )

        elif flatline_fraction >= 0.20:
            level = _lower_trust(
                level,
                "REVIEW",
            )
            reasons.append(
                "Local region contains an elevated "
                "fraction of unchanged samples."
            )

        # This threshold assumes uncalibrated MIT-BIH
        # digital values. It is not portable to mV ECG.
        if local_range < 2.0:
            level = _lower_trust(
                level,
                "UNRELIABLE",
            )
            reasons.append(
                "Local ECG amplitude range is extremely "
                "small in the current digital units."
            )

        score = {
            "HIGH": 1.0,
            "REVIEW": 0.5,
            "UNRELIABLE": 0.0,
        }[level]

        results.append(
            BeatTrustResult(
                sample=sample,
                symbol=str(symbol),
                level=level,
                score=score,
                flatline_fraction=flatline_fraction,
                local_range=local_range,
                reasons=reasons,
            )
        )

    return results
