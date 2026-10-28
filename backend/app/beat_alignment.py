
"""
TRACE — Reference Beat Alignment

Align MIT-BIH reference annotations with
synthetic ECG artefact windows.

Reference annotations are not AI predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.beat_trust import evaluate_beat_trust


ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_PATH = (
    ROOT / "data" / "processed" / "100_annotations_250hz.npz"
)

ARTIFACT_PATH = (
    ROOT / "data" / "processed" / "trace_artifacts.npz"
)


@dataclass
class AlignedBeat:
    absolute_sample: int
    relative_sample: int
    symbol: str
    trust_level: str
    trust_score: float
    flatline_fraction: float
    local_range: float
    reasons: list[str]
    annotation_source: str = "dataset_annotation"
    experimental: bool = True
    clinical_validated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def align_window_beats(
    ecg_window: np.ndarray,
    window_start_sample: int,
    annotation_samples: np.ndarray,
    annotation_symbols: np.ndarray,
    fs: int,
    window_trust: str,
) -> list[dict]:
    """
    Align reference beats to a processed ECG window.

    window_start_sample:
        Zero-based absolute sample index in the
        original resampled ECG recording.
    """

    signal = np.asarray(ecg_window)

    if signal.ndim != 1:
        raise ValueError("ECG window must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("ECG window is too short.")

    if fs <= 0:
        raise ValueError("Invalid sampling frequency.")

    if len(annotation_samples) != len(annotation_symbols):
        raise ValueError(
            "Annotation samples and symbols must match."
        )

    start = int(window_start_sample)
    end = start + len(signal)

    # Select beats in the half-open interval [start, end).
    mask = (
        (annotation_samples >= start)
        & (annotation_samples < end)
    )

    selected_samples = annotation_samples[mask]
    selected_symbols = annotation_symbols[mask]

    relative_samples = (
        selected_samples.astype(np.int64) - start
    )

    if len(relative_samples) == 0:
        return []

    results = evaluate_beat_trust(
        ecg_window=signal,
        beat_samples=relative_samples.tolist(),
        beat_symbols=selected_symbols.tolist(),
        fs=fs,
        window_trust=window_trust,
    )

    aligned: list[dict] = []

    for absolute_sample, result in zip(
        selected_samples,
        results,
    ):
        beat = AlignedBeat(
            absolute_sample=int(absolute_sample),
            relative_sample=int(result.sample),
            symbol=result.symbol,
            trust_level=result.level,
            trust_score=result.score,
            flatline_fraction=result.flatline_fraction,
            local_range=result.local_range,
            reasons=result.reasons,
        )

        aligned.append(beat.to_dict())

    return aligned


def load_reference_annotations() -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(ANNOTATION_PATH)

    with np.load(
        ANNOTATION_PATH,
        allow_pickle=False,
    ) as data:
        samples = data["samples"].copy()
        symbols = data["symbols"].copy()
        fs = int(data["fs"])

        record_id = str(data["record_id"])

    if record_id != "100":
        raise ValueError(
            f"Unexpected annotation record: {record_id}"
        )

    return samples, symbols, fs
