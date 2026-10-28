
from pathlib import Path

import joblib
import numpy as np

from train_artifact_classifier import extract_features

ROOT = Path(__file__).resolve().parents[1]

with np.load(
    ROOT / "data" / "processed" / "trace_artifacts.npz",
    allow_pickle=False,
) as data:
    ecg = data["ecg_corrupted"][:5]
    reference = data["ecg_reference"][:5]
    imu = data["imu_simulated"][:5]
    labels = data["labels"][:5]
    class_names = data["class_names"].tolist()
    fs = int(data["fs"])

bundle = joblib.load(
    ROOT / "models" / "artifact_classifier.joblib"
)
model = bundle["model"]

features = np.asarray(
    [
        extract_features(ecg[i], imu[i], fs)
        for i in range(len(ecg))
    ],
    dtype=np.float32,
)

predictions = model.predict(features)
probabilities = model.predict_proba(features)

print("TRACE CLASSIFIER DIAGNOSTICS")
print("=" * 60)

for i in range(len(labels)):
    difference = ecg[i] - reference[i]

    print(f"\nCLASS: {class_names[int(labels[i])]}")
    print("Predicted:", class_names[int(predictions[i])])
    print("Maximum probability:", round(float(max(probabilities[i])), 4))
    print("ECG standard deviation:", round(float(np.std(ecg[i])), 3))
    print("ECG peak-to-peak:", round(float(np.ptp(ecg[i])), 3))
    print("Injected difference RMS:",
          round(float(np.sqrt(np.mean(difference ** 2))), 3))
    print("IMU standard deviation:", round(float(np.std(imu[i])), 4))

print("\nFEATURE IMPORTANCE")
importance = model.feature_importances_
for index in np.argsort(importance)[::-1]:
    print(f"Feature {index}: {importance[index]}")
