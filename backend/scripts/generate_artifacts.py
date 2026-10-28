
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "100_250hz.npz"
OUTPUT = ROOT / "data" / "processed" / "trace_artifacts.npz"

FS = 250
WINDOW_SECONDS = 4
WINDOW_SIZE = FS * WINDOW_SECONDS
SEED = 42

CLASSES = [
    "clean",
    "baseline_wander",
    "walking",
    "electrode_pop",
    "lead_dropout",
]


def simulate_artifact(clean, label, rng):
    n = len(clean)
    t = np.arange(n) / FS

    corrupted = clean.copy()
    imu = rng.normal(0.0, 0.015, n)

    # Amplitudes are in the same uncalibrated digital
    # units as the recorded ECG.
    if label == "baseline_wander":
        frequency = rng.uniform(0.12, 0.45)
        amplitude = rng.uniform(25, 75)
        phase = rng.uniform(0, 2 * np.pi)

        corrupted += amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )

    elif label == "walking":
        frequency = rng.uniform(1.2, 2.4)
        amplitude = rng.uniform(30, 90)
        phase = rng.uniform(0, 2 * np.pi)

        motion = np.sin(
            2 * np.pi * frequency * t + phase
        )

        corrupted += amplitude * motion
        corrupted += rng.normal(0, 12, n)

        imu += 0.45 * motion

    elif label == "electrode_pop":
        start = int(rng.integers(100, n - 150))
        duration = int(rng.integers(40, 120))
        end = min(start + duration, n)

        amplitude = rng.uniform(120, 260)
        direction = rng.choice([-1, 1])

        decay = np.exp(
            -np.linspace(0, 4, end - start)
        )

        corrupted[start:end] += (
            direction * amplitude * decay
        )

        imu[start:end] += rng.normal(
            0.3, 0.08, end - start
        )

    elif label == "lead_dropout":
        start = int(rng.integers(100, n - 200))
        duration = int(rng.integers(80, 180))
        end = min(start + duration, n)

        baseline = np.median(clean)
        corrupted[start:end] = (
            baseline
            + rng.normal(0, 1.0, end - start)
        )

        # A dropout does not necessarily produce motion.
        # IMU remains close to its simulated noise floor.

    return corrupted, imu


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with np.load(INPUT, allow_pickle=False) as data:
        ecg = data["ecg"][:, 0].astype(np.float32)
        fs = int(data["fs"])

    if fs != FS:
        raise ValueError(f"Expected {FS} Hz, got {fs} Hz")

    rng = np.random.default_rng(SEED)

    windows = []
    corrupted_windows = []
    imu_windows = []
    labels = []
    start_samples = []

    # Non-overlapping windows avoid duplicated source
    # samples in this initial experiment.
    for start in range(0, len(ecg) - WINDOW_SIZE + 1, WINDOW_SIZE):
        clean = ecg[start:start + WINDOW_SIZE].copy()

        if not np.isfinite(clean).all():
            continue

        for label_id, label in enumerate(CLASSES):
            corrupted, imu = simulate_artifact(
                clean.astype(np.float64),
                label,
                rng,
            )

            windows.append(clean)
            corrupted_windows.append(
                corrupted.astype(np.float32)
            )
            imu_windows.append(imu.astype(np.float32))
            labels.append(label_id)
            start_samples.append(start)

    if not windows:
        raise RuntimeError("No valid ECG windows generated.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        ecg_reference=np.stack(windows),
        ecg_corrupted=np.stack(corrupted_windows),
        imu_simulated=np.stack(imu_windows),
        labels=np.asarray(labels, dtype=np.int32),
        class_names=np.asarray(CLASSES),
        start_samples=np.asarray(start_samples, dtype=np.int64),
        fs=np.int32(FS),
        window_seconds=np.int32(WINDOW_SECONDS),
        seed=np.int32(SEED),
        record_id=np.array("100"),
        units=np.array("uncalibrated_digital_values"),
    )

    print("TRACE ARTIFACT DATASET GENERATED")
    print("Source: MIT-BIH Record 100")
    print("ECG: Recorded, resampled to 250 Hz")
    print("Artifacts: SIMULATED")
    print("IMU: SIMULATED")
    print("Classes:", CLASSES)
    print("Windows per class:", len(windows) // len(CLASSES))
    print("Total labeled windows:", len(labels))
    print("Samples per window:", WINDOW_SIZE)
    print("Dataset shape:", (len(labels), WINDOW_SIZE))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
