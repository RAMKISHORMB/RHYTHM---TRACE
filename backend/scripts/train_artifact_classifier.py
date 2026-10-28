
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.signal import welch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# TRACE — AI ARTEFACT CLASSIFIER
# Research prototype | MEDAITHON'26
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "trace_artifacts.npz"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "artifact_classifier.joblib"

SEED = 42

FEATURE_NAMES = [
    "ecg_std",
    "ecg_peak_to_peak",
    "ecg_rms_centered",
    "ecg_mean_abs_derivative",
    "ecg_derivative_std",
    "ecg_max_abs_derivative",
    "ecg_bandpower_0_05_0_5",
    "ecg_bandpower_0_5_5",
    "ecg_bandpower_5_40",
    "imu_std",
    "imu_rms",
    "imu_max_abs",
    "imu_mean_abs_derivative",
    "ecg_min_local_std",
    "ecg_p10_local_std",
    "ecg_min_local_range",
    "ecg_flat_segment_fraction",
    "ecg_longest_flat_run",
    "ecg_block_median_range",
]


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def extract_features(ecg, imu, fs):
    """
    Extract 19 ECG and simulated-IMU features
    from a four-second signal window.

    ECG values are uncalibrated digital values.
    IMU measurements are simulated.
    """

    ecg = np.asarray(ecg, dtype=np.float64)
    imu = np.asarray(imu, dtype=np.float64)

    if ecg.ndim != 1 or imu.ndim != 1:
        raise ValueError("ECG and IMU must be one-dimensional.")

    if len(ecg) != len(imu):
        raise ValueError("ECG and IMU lengths must match.")

    if len(ecg) != 4 * fs:
        raise ValueError("Expected a four-second window.")

    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains non-finite values.")

    if not np.isfinite(imu).all():
        raise ValueError("IMU contains non-finite values.")

    centered = ecg - np.median(ecg)
    derivative = np.diff(ecg)

    # --------------------------------------------------------
    # Frequency-domain ECG features
    # --------------------------------------------------------

    frequencies, power = welch(
        centered,
        fs=fs,
        nperseg=min(512, len(ecg)),
    )

    def bandpower(low, high):
        mask = (frequencies >= low) & (frequencies < high)

        if np.count_nonzero(mask) < 2:
            return 0.0

        return float(
            np.trapezoid(
                power[mask],
                frequencies[mask],
            )
        )

    # --------------------------------------------------------
    # Local flatline / lead-dropout features
    # --------------------------------------------------------

    segment_size = max(2, int(0.1 * fs))
    segment_count = len(ecg) // segment_size

    segments = ecg[
        :segment_count * segment_size
    ].reshape(segment_count, segment_size)

    local_std = np.std(segments, axis=1)
    local_range = np.ptp(segments, axis=1)

    # Experimental threshold in digital units.
    flat_mask = local_std < 2.0

    longest_flat_run = 0
    current_run = 0

    for is_flat in flat_mask:
        if is_flat:
            current_run += 1
            longest_flat_run = max(
                longest_flat_run,
                current_run,
            )
        else:
            current_run = 0

    # --------------------------------------------------------
    # Baseline drift feature
    # --------------------------------------------------------

    block_size = max(2, int(0.5 * fs))
    block_count = len(ecg) // block_size

    block_medians = np.median(
        ecg[:block_count * block_size].reshape(
            block_count,
            block_size,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = [
        float(np.std(ecg)),
        float(np.ptp(ecg)),
        rms(centered),
        float(np.mean(np.abs(derivative))),
        float(np.std(derivative)),
        float(np.max(np.abs(derivative))),
        bandpower(0.05, 0.5),
        bandpower(0.5, 5.0),
        bandpower(5.0, 40.0),
        float(np.std(imu)),
        rms(imu),
        float(np.max(np.abs(imu))),
        float(np.mean(np.abs(np.diff(imu)))),
        float(np.min(local_std)),
        float(np.percentile(local_std, 10)),
        float(np.min(local_range)),
        float(np.mean(flat_mask)),
        float(longest_flat_run),
        float(np.ptp(block_medians)),
    ]

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("Feature count mismatch.")

    return features


def main():

    # --------------------------------------------------------
    # 1. Load the generated artefact dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        ecg = data["ecg_corrupted"]
        imu = data["imu_simulated"]
        labels = data["labels"]
        groups = data["start_samples"]
        class_names = data["class_names"].tolist()
        fs = int(data["fs"])

    if fs != 250:
        raise ValueError(f"Unexpected sampling rate: {fs}")

    if not (
        len(ecg) == len(imu) == len(labels) == len(groups)
    ):
        raise ValueError("Dataset array lengths do not match.")

    print("=" * 60)
    print("TRACE — LIGHTGBM ARTEFACT CLASSIFIER")
    print("=" * 60)

    print("Dataset windows:", len(labels))
    print("Classes:", class_names)
    print("Features per window:", len(FEATURE_NAMES))
    print("Sampling frequency:", fs, "Hz")

    # --------------------------------------------------------
    # 2. Extract features
    # --------------------------------------------------------

    print("\nExtracting ECG and simulated IMU features...")

    features = np.asarray(
        [
            extract_features(ecg[i], imu[i], fs)
            for i in range(len(labels))
        ],
        dtype=np.float32,
    )

    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains invalid values.")

    print("Feature matrix shape:", features.shape)

    # --------------------------------------------------------
    # 3. Group-safe train/test split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            features,
            labels,
            groups=groups,
        )
    )

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError(
            "Data leakage: source window appears "
            "in both training and testing."
        )

    print("\nTraining windows:", len(train_idx))
    print("Testing windows:", len(test_idx))
    print("Source-window leakage: NONE")

    # --------------------------------------------------------
    # 4. Train LightGBM
    # --------------------------------------------------------

    print("\nTraining LightGBM...")

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        random_state=SEED,
        verbosity=-1,
    )

    model.fit(
        features[train_idx],
        labels[train_idx],
    )

    # --------------------------------------------------------
    # 5. Evaluate model
    # --------------------------------------------------------

    predictions = model.predict(features[test_idx])

    accuracy = accuracy_score(
        labels[test_idx],
        predictions,
    )

    print("\n" + "=" * 60)
    print("TRACE — CLASSIFIER EVALUATION")
    print("=" * 60)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            labels[test_idx],
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            labels[test_idx],
            predictions,
            labels=list(range(len(class_names))),
        )
    )

    # --------------------------------------------------------
    # 6. Feature importance
    # --------------------------------------------------------

    print("\nTop 10 important features:")

    importance = model.feature_importances_

    ranked_indices = np.argsort(
        importance
    )[::-1][:10]

    for rank, index in enumerate(ranked_indices, start=1):
        print(
            f"{rank:2d}. "
            f"{FEATURE_NAMES[index]:32s} "
            f"{importance[index]}"
        )

    # --------------------------------------------------------
    # 7. Save trained model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": model,
        "class_names": class_names,
        "fs": fs,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "window_seconds": 4,
        "training_source": "MIT-BIH record 100",
        "artifact_source": "simulated",
        "imu_source": "simulated",
        "random_seed": SEED,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("\nModel saved:", MODEL_PATH)

    print("\nIMPORTANT:")
    print(
        "This is experimental classification of "
        "simulated ECG artefacts."
    )
    print(
        "The reported accuracy does not establish "
        "clinical safety or real-world generalization."
    )


if __name__ == "__main__":
    main()
