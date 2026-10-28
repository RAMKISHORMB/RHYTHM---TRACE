
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "mitdb" / "100annotations.txt"
ECG_FILE = ROOT / "data" / "processed" / "100_250hz.npz"
OUTPUT = ROOT / "data" / "processed" / "100_annotations_250hz.npz"

ORIGINAL_FS = 360
TARGET_FS = 250

# MIT-BIH beat annotation symbols.
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not ECG_FILE.exists():
        raise FileNotFoundError(ECG_FILE)

    with np.load(ECG_FILE) as data:
        signal_length = len(data["ecg"])
        fs = int(data["fs"])

    if fs != TARGET_FS:
        raise ValueError(f"Unexpected processed ECG rate: {fs}")

    original_samples = []
    converted_samples = []
    symbols = []
    skipped = 0

    with INPUT.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.lower().startswith("time"):
                continue

            # Example:
            # 0:00.447  161  N  0  0  0
            parts = re.split(r"\s+", line)

            if len(parts) < 3:
                raise ValueError(
                    f"Malformed annotation at line {line_number}: {line}"
                )

            try:
                sample = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid sample at line {line_number}: {line}"
                ) from exc

            symbol = parts[2]

            if symbol not in BEAT_SYMBOLS:
                skipped += 1
                continue

            converted = (sample * TARGET_FS * 2 + ORIGINAL_FS) // (
                2 * ORIGINAL_FS
            )

            if not 0 <= converted < signal_length:
                raise ValueError(
                    f"Annotation outside ECG range: {sample}"
                )

            original_samples.append(sample)
            converted_samples.append(converted)
            symbols.append(symbol)

    original_samples = np.asarray(original_samples, dtype=np.int64)
    converted_samples = np.asarray(converted_samples, dtype=np.int64)
    symbols = np.asarray(symbols, dtype="U2")

    if len(converted_samples) == 0:
        raise ValueError("No beat annotations found.")

    if not np.all(np.diff(original_samples) > 0):
        raise ValueError("Original beat annotations are not increasing.")

    if not np.all(np.diff(converted_samples) > 0):
        raise ValueError("Resampled beat positions are not increasing.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        samples=converted_samples,
        original_samples=original_samples,
        symbols=symbols,
        fs=np.int32(TARGET_FS),
        record_id=np.array("100"),
    )

    print("TRACE ANNOTATION PREPROCESSING COMPLETE")
    print("Beat annotations:", len(converted_samples))
    print("Non-beat annotations skipped:", skipped)
    print("Original frequency:", ORIGINAL_FS, "Hz")
    print("Target frequency:", TARGET_FS, "Hz")
    print("First 5 original samples:", original_samples[:5])
    print("First 5 converted samples:", converted_samples[:5])
    print("First 5 beat symbols:", symbols[:5].tolist())
    print("Last beat time:", round(converted_samples[-1] / TARGET_FS, 3))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
