
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt


TrustLevel = Literal["HIGH", "REVIEW", "UNRELIABLE"]

CleaningStatus = Literal[
    "PRESERVED",
    "CANDIDATE_FILTERED",
    "ABSTAINED",
]


@dataclass
class CleaningResult:
    raw_ecg: np.ndarray
    cleaned_ecg: np.ndarray | None
    status: CleaningStatus
    method: str
    reasons: list[str]
    experimental: bool = True
    clinical_validated: bool = False


def _baseline_filter(
    ecg: np.ndarray,
    fs: int,
    cutoff_hz: float = 0.5,
) -> np.ndarray:
    """
    Experimental offline zero-phase high-pass filter.

    Preserve an estimated digital baseline obtained
    exclusively from the corrupted input signal.

    This does not reconstruct the true original baseline.
    """

    sos = butter(
        N=2,
        Wn=cutoff_hz,
        btype="highpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, ecg)

    # Restore a baseline estimated from the input,
    # never from the unavailable clean reference.
    estimated_baseline = float(np.median(ecg))

    return filtered + estimated_baseline


def clean_ecg_window(
    ecg_window: np.ndarray,
    fs: int,
    predicted_artifact: str,
    trust_level: TrustLevel,
) -> CleaningResult:
    """
    Conservative artefact-aware ECG cleaning policy.

    HIGH:
        Preserve the raw waveform.

    REVIEW + baseline_wander:
        Generate an experimental filtered candidate.

    REVIEW + other artefacts:
        Preserve without filtering.

    UNRELIABLE:
        Abstain from cleaning.
    """

    raw = np.asarray(
        ecg_window,
        dtype=np.float64,
    ).copy()

    if raw.ndim != 1:
        raise ValueError(
            "ECG window must be one-dimensional."
        )

    if raw.size == 0:
        raise ValueError(
            "ECG window cannot be empty."
        )

    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "ECG contains NaN or infinite values."
        )

    if fs <= 0:
        raise ValueError(
            "Sampling rate must be positive."
        )

    if trust_level not in (
        "HIGH",
        "REVIEW",
        "UNRELIABLE",
    ):
        raise ValueError(
            "Invalid trust level."
        )

    if trust_level == "UNRELIABLE":
        return CleaningResult(
            raw_ecg=raw,
            cleaned_ecg=None,
            status="ABSTAINED",
            method="none",
            reasons=[
                "Window trust is UNRELIABLE.",
                "No waveform reconstruction attempted.",
            ],
        )

    if (
        trust_level == "REVIEW"
        and predicted_artifact == "baseline_wander"
    ):
        if raw.size < max(30, fs):
            return CleaningResult(
                raw_ecg=raw,
                cleaned_ecg=None,
                status="ABSTAINED",
                method="none",
                reasons=[
                    "Window too short for experimental "
                    "baseline filtering."
                ],
            )

        candidate = _baseline_filter(
            ecg=raw,
            fs=fs,
        )

        if not np.all(np.isfinite(candidate)):
            return CleaningResult(
                raw_ecg=raw,
                cleaned_ecg=None,
                status="ABSTAINED",
                method="none",
                reasons=[
                    "Filtering produced invalid values."
                ],
            )

        return CleaningResult(
            raw_ecg=raw,
            cleaned_ecg=candidate,
            status="CANDIDATE_FILTERED",
            method=(
                "butterworth_highpass_0.5hz_"
                "median_baseline_restored"
            ),
            reasons=[
                "Baseline wander predicted.",
                "Window trust is REVIEW.",
                "Zero-phase high-pass filtering applied.",
                "Input-derived median baseline restored.",
                "Output is experimental and must be "
                "evaluated before downstream use.",
            ],
        )

    return CleaningResult(
        raw_ecg=raw,
        cleaned_ecg=raw.copy(),
        status="PRESERVED",
        method="identity",
        reasons=[
            "Current policy does not permit filtering."
        ],
    )
