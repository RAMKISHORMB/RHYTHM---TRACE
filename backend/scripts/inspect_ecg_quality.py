
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ecg_quality import analyze_ecg_quality
from app.motion_intelligence import analyze_motion


def main():
    path = ROOT / "data" / "processed" / "trace_artifacts.npz"

    with np.load(path, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        imu = data["imu_simulated"]
        labels = data["labels"]
        names = data["class_names"]
        fs = int(data["fs"])

    print("\nTRACE — ECG + MOTION QUALITY INSPECTION")
    print("=" * 75)

    for class_id, name in enumerate(names):
        index = int(np.where(labels == class_id)[0][0])

        quality = analyze_ecg_quality(ecg[index], fs=fs)
        motion = analyze_motion(imu[index])

        print(f"\nCLASS: {name}")
        print(f"Baseline ratio: {quality.baseline_ratio:.4f}")
        print(
            f"Abrupt-change ratio: "
            f"{quality.step_to_median_ratio:.4f}"
        )
        print(
            f"Longest flat interval: "
            f"{quality.longest_flat_duration_ms:.1f} ms"
        )
        print(f"Motion RMS: {motion.rms:.5f}")
        print(f"Motion level: {motion.activity_level}")
        print(f"ECG quality flags: {quality.quality_flags}")

    print("\nINSPECTION COMPLETE")
    print("Research prototype; thresholds not clinically validated.")


if __name__ == "__main__":
    main()
