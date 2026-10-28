
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.beat_alignment import (
    ARTIFACT_PATH,
    align_window_beats,
    load_reference_annotations,
)


def main():
    annotation_samples, annotation_symbols, annotation_fs = (
        load_reference_annotations()
    )

    with np.load(
        ARTIFACT_PATH,
        allow_pickle=False,
    ) as data:
        starts = data["start_samples"][:5].copy()
        windows = data["ecg_corrupted"][:5].copy()
        fs = int(data["fs"])

    assert fs == annotation_fs == 250

    total_beats = 0

    for index, (start, window) in enumerate(
        zip(starts, windows),
        start=1,
    ):
        beats = align_window_beats(
            ecg_window=window,
            window_start_sample=int(start),
            annotation_samples=annotation_samples,
            annotation_symbols=annotation_symbols,
            fs=fs,
            window_trust="REVIEW",
        )

        total_beats += len(beats)

        print(
            f"Window {index}: "
            f"start={int(start)} | "
            f"aligned beats={len(beats)}"
        )

        for beat in beats:
            assert (
                beat["absolute_sample"]
                == int(start) + beat["relative_sample"]
            )

            assert (
                0 <= beat["relative_sample"] < len(window)
            )

            print(
                f"  Beat {beat['symbol']} | "
                f"absolute={beat['absolute_sample']} | "
                f"relative={beat['relative_sample']} | "
                f"trust={beat['trust_level']}"
            )

    assert total_beats > 0, (
        "No reference beats were aligned. "
        "Check start_samples and annotation provenance."
    )

    print(
        f"\nTotal aligned beats: {total_beats}"
    )

    print("TRACE BEAT ALIGNMENT TEST PASSED")


if __name__ == "__main__":
    main()
