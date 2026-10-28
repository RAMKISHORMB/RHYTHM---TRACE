
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.train_artifact_classifier import (
    FEATURE_NAMES,
    extract_features,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "artifact_classifier.joblib"


class ArtifactInference:
    """
    TRACE experimental ECG artefact classifier.

    Input:
        Four-second ECG window at 250 Hz.
        Corresponding simulated IMU window.

    Output:
        Predicted artefact class and model probabilities.

    Not validated for clinical use.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"TRACE model not found: {MODEL_PATH}"
            )

        bundle = joblib.load(MODEL_PATH)

        self.model = bundle["model"]
        self.class_names = list(bundle["class_names"])
        self.fs = int(bundle["fs"])
        self.feature_count = int(bundle["feature_count"])
        self.window_seconds = int(
            bundle.get("window_seconds", 4)
        )

        saved_feature_names = bundle.get("feature_names")

        if self.fs != 250:
            raise ValueError("Expected a 250 Hz model.")

        if self.feature_count != len(FEATURE_NAMES):
            raise ValueError(
                "Model feature count does not match extractor."
            )

        if (
            saved_feature_names is not None
            and list(saved_feature_names) != FEATURE_NAMES
        ):
            raise ValueError(
                "Model feature ordering does not match extractor."
            )

        if self.window_seconds != 4:
            raise ValueError("Expected a four-second model.")

        print(
            "TRACE artefact classifier loaded:",
            MODEL_PATH.name,
        )

    def predict(
        self,
        ecg_window: np.ndarray,
        imu_window: np.ndarray,
    ) -> dict[str, Any]:

        ecg = np.asarray(
            ecg_window,
            dtype=np.float64,
        )

        imu = np.asarray(
            imu_window,
            dtype=np.float64,
        )

        expected_length = self.fs * self.window_seconds

        if ecg.shape != (expected_length,):
            raise ValueError(
                f"Expected ECG shape ({expected_length},), "
                f"received {ecg.shape}"
            )

        if imu.shape != (expected_length,):
            raise ValueError(
                f"Expected IMU shape ({expected_length},), "
                f"received {imu.shape}"
            )

        if not np.isfinite(ecg).all():
            raise ValueError("ECG contains invalid values.")

        if not np.isfinite(imu).all():
            raise ValueError("IMU contains invalid values.")

        features = np.asarray(
            extract_features(ecg, imu, self.fs),
            dtype=np.float32,
        ).reshape(1, -1)

        if features.shape[1] != self.feature_count:
            raise ValueError("Feature count mismatch.")

        predicted_id = int(
            self.model.predict(features)[0]
        )

        probabilities = self.model.predict_proba(
            features
        )[0]

        # Match probabilities to the classifier's
        # actual class ordering.
        probability_by_class = {
            self.class_names[int(class_id)]: float(probability)
            for class_id, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

        predicted_class = self.class_names[predicted_id]

        return {
            "predicted_class": predicted_class,
            "predicted_class_id": predicted_id,
            "model_probability": probability_by_class[
                predicted_class
            ],
            "class_probabilities": probability_by_class,
            "model_type": "LightGBM",
            "feature_count": self.feature_count,
            "experimental": True,
            "clinical_validated": False,
        }
