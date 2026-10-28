
from pathlib import Path
from collections import Counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.arrhythmia_features import (
    FEATURE_NAMES,
    extract_beat_features,
)


def main():
    processed = ROOT / "data" / "processed"

    with np.load(
        processed / "100_250hz.npz",
        allow_pickle=False,
    ) as data:
        ecg = data["ecg"][:, 0].copy()
        fs = int(data["fs"])

    with np.load(
        processed / "100_annotations_250hz.npz",
        allow_pickle=False,
    ) as data:
        samples = data["samples"].copy()
        symbols = data["symbols"].copy()
        annotation_fs = int(data["fs"])

    if fs != annotation_fs:
        raise ValueError("Sampling rates do not match.")

    print("TRACE — ARRHYTHMIA FEATURE INSPECTION")
    print("Sampling rate:", fs)
    print("ECG samples:", len(ecg))
    print("Annotations:", len(samples))
    print("Annotation symbols:", dict(Counter(symbols)))

    # Exclude non-beat annotations from classification.
    beat_symbols = {
        "N", "L", "R", "A", "a", "J", "S",
        "V", "F", "e", "j", "E", "/",
        "f", "Q",
    }

    mask = np.isin(symbols, list(beat_symbols))

    beat_samples = samples[mask]
    beat_labels = symbols[mask]

    features, valid_indices = extract_beat_features(
        ecg=ecg,
        beat_samples=beat_samples,
        fs=fs,
    )

    valid_labels = beat_labels[valid_indices]

    assert features.shape[1] == len(FEATURE_NAMES)
    assert np.all(np.isfinite(features))
    assert len(features) == len(valid_labels)

    print()
    print("Beat annotations:", len(beat_samples))
    print("Valid feature rows:", len(features))
    print("Feature count:", features.shape[1])
    print("Feature matrix shape:", features.shape)
    print("Valid label distribution:")
    print(dict(Counter(valid_labels)))

    print()
    print("First feature row:")
    for name, value in zip(
        FEATURE_NAMES,
        features[0],
    ):
        print(f"{name}: {value:.4f}")

    print()
    print("FEATURE EXTRACTION PASSED")


if __name__ == "__main__":
    main()
