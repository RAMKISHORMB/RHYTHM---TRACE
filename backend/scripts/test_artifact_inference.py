
from pathlib import Path

import joblib
import numpy as np

from train_artifact_classifier import extract_features

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "artifact_classifier.joblib"
DATA_PATH = ROOT / "data" / "processed" / "trace_artifacts.npz"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    class_names = bundle["class_names"]
    fs = int(bundle["fs"])

    with np.load(DATA_PATH, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        imu = data["imu_simulated"]
        labels = data["labels"]
        data_fs = int(data["fs"])

    if data_fs != fs:
        raise ValueError("Sampling frequency mismatch.")

    print("TRACE AI INFERENCE TEST")
    print("Model:", MODEL_PATH.name)
    print("Sampling rate:", fs, "Hz")
    print()

    # Test one example from each class.
    for index in range(len(class_names)):
        features = np.asarray(
            extract_features(ecg[index], imu[index], fs),
            dtype=np.float32,
        ).reshape(1, -1)

        if features.shape[1] != bundle["feature_count"]:
            raise ValueError("Feature count mismatch.")

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        predicted_name = class_names[prediction]
        actual_name = class_names[int(labels[index])]
        confidence = float(np.max(probabilities))

        print("Expected:", actual_name)
        print("Predicted:", predicted_name)
        print("Model probability:", f"{confidence:.2%}")
        print("-" * 40)

    print("Inference pipeline executed successfully.")
    print(
        "Note: These examples come from the dataset used "
        "for training and evaluation; this is a smoke test, "
        "not an independent performance measurement."
    )


if __name__ == "__main__":
    main()
