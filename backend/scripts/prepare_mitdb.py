
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "mitdb" / "100.csv"
OUTPUT = ROOT / "data" / "processed" / "100_250hz.npz"

ORIGINAL_FS = 360
TARGET_FS = 250

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing ECG file: {INPUT}")

    df = pd.read_csv(INPUT)
    df.columns = df.columns.str.strip().str.strip("'\"")

    required = {"sample #", "MLII", "V5"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    samples = df["sample #"].to_numpy(dtype=np.int64)

    if not np.array_equal(samples, np.arange(len(samples))):
        raise ValueError("Sample numbers are not continuous from zero.")

    ecg = df[["MLII", "V5"]].to_numpy(dtype=np.float64)

    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains invalid or missing values.")

    # Convert 360 Hz to 250 Hz using anti-aliasing filtering.
    processed = resample_poly(ecg, 25, 36, axis=0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        ecg=processed.astype(np.float32),
        fs=np.int32(TARGET_FS),
        original_fs=np.int32(ORIGINAL_FS),
        channels=np.array(["MLII", "V5"]),
        record_id=np.array("100"),
        units=np.array("uncalibrated_digital_values"),
        source=np.array("MIT-BIH Arrhythmia Database Kaggle CSV export"),
    )

    print("TRACE ECG PREPROCESSING COMPLETE")
    print("Original samples:", len(ecg))
    print("Original frequency:", ORIGINAL_FS, "Hz")
    print("Processed samples:", len(processed))
    print("Target frequency:", TARGET_FS, "Hz")
    print("Original duration:", round(len(ecg) / ORIGINAL_FS, 3), "seconds")
    print("Processed duration:", round(len(processed) / TARGET_FS, 3), "seconds")
    print("Processed shape:", processed.shape)
    print("Output:", OUTPUT)

if __name__ == "__main__":
    main()
