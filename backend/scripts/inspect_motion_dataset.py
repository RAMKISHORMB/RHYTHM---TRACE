
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.motion_intelligence import analyze_motion


def main():
    path = ROOT / "data" / "processed" / "trace_artifacts.npz"

    with np.load(path, allow_pickle=False) as data:
        imu = data["imu_simulated"]
        labels = data["labels"]
        class_names = data["class_names"]

    scores = np.array([
        analyze_motion(window).activity_score
        for window in imu
    ])

    print("\nTRACE — MOTION DATASET ANALYSIS")
    print("=" * 68)
    print("Windows:", len(scores))
    print("IMU shape:", imu.shape)
    print()

    print(
        f"{'CLASS':<20}"
        f"{'COUNT':>8}"
        f"{'MEAN':>12}"
        f"{'P50':>12}"
        f"{'P95':>12}"
    )
    print("-" * 68)

    for class_id, name in enumerate(class_names):
        values = scores[labels == class_id]

        print(
            f"{str(name):<20}"
            f"{len(values):>8}"
            f"{np.mean(values):>12.5f}"
            f"{np.percentile(values, 50):>12.5f}"
            f"{np.percentile(values, 95):>12.5f}"
        )

    print("\nOVERALL MOTION RMS PERCENTILES")

    for percentile in (5, 25, 50, 75, 95, 99):
        print(
            f"P{percentile}: "
            f"{np.percentile(scores, percentile):.6f}"
        )

    print("\nCURRENT HEURISTIC CATEGORIES")

    for level in ("LOW", "MODERATE", "HIGH"):
        if level == "LOW":
            count = np.sum(scores < 0.05)
        elif level == "MODERATE":
            count = np.sum(
                (scores >= 0.05) & (scores < 0.25)
            )
        else:
            count = np.sum(scores >= 0.25)

        print(f"{level}: {count} windows")

    print("\nResearch note:")
    print("Motion values come from a simulated 1-channel proxy.")
    print("No physical activity or clinical validation is implied.")


if __name__ == "__main__":
    main()
