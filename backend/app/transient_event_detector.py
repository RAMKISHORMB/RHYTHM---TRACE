
"""
TRACE — Transient Cardiac-Event Candidate Detector
MEDAITHON'26 | Cardiology PS-001

Detects unusual RR timing patterns from ECG.

This is a research prototype, not a diagnostic
arrhythmia detector. No reference annotations are
used during inference.
"""

from dataclasses import dataclass, asdict

import numpy as np

from app.rpeak_detector import detect_rpeaks


@dataclass
class EventCandidate:
    event_type: str
    timestamp_seconds: float
    rr_interval_seconds: float
    baseline_rr_seconds: float
    ratio_to_baseline: float
    explanation: str
    requires_review: bool = True


@dataclass
class TransientEventResult:
    peak_count: int
    estimated_heart_rate_bpm: float | None
    event_count: int
    events: list[dict]
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self):
        return asdict(self)


def detect_transient_events(
    ecg,
    fs=250,
    short_rr_ratio=0.80,
    long_rr_ratio=1.25,
    baseline_beats=8,
):
    """
    Flag RR intervals that differ substantially
    from a preceding local RR baseline.

    The thresholds are provisional and have not
    been validated on independent patient records.
    """

    if not isinstance(baseline_beats, int) or baseline_beats < 3:
        raise ValueError(
            "baseline_beats must be an integer >= 3."
        )

    if not (
        np.isfinite(short_rr_ratio)
        and np.isfinite(long_rr_ratio)
        and 0 < short_rr_ratio < 1 < long_rr_ratio
    ):
        raise ValueError("Invalid RR ratio thresholds.")

    peak_result = detect_rpeaks(ecg, fs=fs)

    peak_times = np.asarray(
        peak_result.peak_times_seconds,
        dtype=np.float64,
    )

    rr = np.diff(peak_times)

    events = []

    # Use only preceding intervals for each baseline.
    # This avoids looking ahead to future beats.
    for i in range(baseline_beats, len(rr)):
        previous_rr = rr[
            i - baseline_beats:i
        ]

        baseline = float(
            np.median(previous_rr)
        )

        current_rr = float(rr[i])

        if baseline <= 0:
            continue

        ratio = current_rr / baseline

        # Timestamp corresponds to the beat ending
        # the unusual RR interval.
        timestamp = float(peak_times[i + 1])

        if ratio < short_rr_ratio:
            event_type = "SHORT_RR_CANDIDATE"
            explanation = (
                f"RR interval {current_rr:.3f}s is "
                f"{ratio:.2f}x the preceding local "
                f"baseline of {baseline:.3f}s."
            )

        elif ratio > long_rr_ratio:
            event_type = "LONG_RR_CANDIDATE"
            explanation = (
                f"RR interval {current_rr:.3f}s is "
                f"{ratio:.2f}x the preceding local "
                f"baseline of {baseline:.3f}s."
            )

        else:
            continue

        events.append(
            asdict(
                EventCandidate(
                    event_type=event_type,
                    timestamp_seconds=timestamp,
                    rr_interval_seconds=current_rr,
                    baseline_rr_seconds=baseline,
                    ratio_to_baseline=float(ratio),
                    explanation=explanation,
                )
            )
        )

    return TransientEventResult(
        peak_count=peak_result.peak_count,
        estimated_heart_rate_bpm=(
            peak_result.estimated_heart_rate_bpm
        ),
        event_count=len(events),
        events=events,
    )
