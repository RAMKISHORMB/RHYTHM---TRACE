
"""
TRACE — Hybrid Pipeline Integration Test

Uses one synthetic ECG window from each artefact
class. Ground-truth labels are used only to select
and identify test examples, never as predictions.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hybrid_pipeline import HybridPipeline


def main():
    dataset_path = (
        ROOT / "data" / "processed"
        / "trace_artifacts.npz"
    )

    with np.load(dataset_path, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        imu = data["imu_simulated"]
        labels = data["labels"]
        class_names = data["class_names"]

    pipeline = HybridPipeline()

    print("\nTRACE — HYBRID PIPELINE TEST")
    print("=" * 85)

    for class_id, class_name in enumerate(class_names):
        index = int(
            np.where(labels == class_id)[0][0]
        )

        result = pipeline.analyze(
            ecg[index],
            imu[index],
        )

        print(f"\nSCENARIO: {class_name}")
        print(
            "LightGBM prediction:",
            result.artifact["predicted_class"],
        )
        print(
            "Model probability:",
            round(
                result.artifact["model_probability"],
                4,
            ),
        )
        print(
            "Hybrid decision:",
            result.fusion["decision"],
        )
        print(
            "Evidence conflict:",
            result.model_disagreement,
        )
        print(
            "Requires review:",
            result.requires_review,
        )
        print(
            "Quality flags:",
            result.fusion["ecg_quality_flags"],
        )
        print(
            "Dropout duration:",
            result.fusion["dropout_duration_ms"],
            "ms",
        )

    print("\nHYBRID PIPELINE INTEGRATION COMPLETE")
    print(
        "Synthetic data only; not clinically validated."
    )


if __name__ == "__main__":
    main()
