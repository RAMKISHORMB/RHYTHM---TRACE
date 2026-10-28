
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from app.artifact_inference import ArtifactInference


DATA_PATH = ROOT / "data" / "processed" / "trace_artifacts.npz"


def main():
    classifier = ArtifactInference()

    with np.load(DATA_PATH, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"][:5]
        imu = data["imu_simulated"][:5]
        labels = data["labels"][:5]
        class_names = data["class_names"].tolist()

    print("\nTRACE — BACKEND AI MODULE TEST")
    print("=" * 50)

    correct = 0

    for index in range(5):
        result = classifier.predict(
            ecg[index],
            imu[index],
        )

        actual = class_names[int(labels[index])]
        predicted = result["predicted_class"]

        matched = actual == predicted

        if matched:
            correct += 1

        print(f"\nExample {index + 1}")
        print("Expected:", actual)
        print("Predicted:", predicted)
        print(
            "Model probability:",
            f"{result['model_probability']:.2%}",
        )
        print("Match:", matched)

    print("\n" + "=" * 50)
    print("Correct examples:", f"{correct}/5")
    print("AI module integration test completed.")

    print(
        "\nThese five examples are a smoke test, "
        "not an independent accuracy evaluation."
    )


if __name__ == "__main__":
    main()
