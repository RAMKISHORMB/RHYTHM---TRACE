
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "trace_artifacts.npz"
OUTPUT = ROOT / "data" / "processed" / "artifact_comparison.png"

with np.load(INPUT, allow_pickle=False) as data:
    reference = data["ecg_reference"][:5]
    corrupted = data["ecg_corrupted"][:5]
    imu = data["imu_simulated"][:5]
    labels = data["labels"][:5]
    class_names = data["class_names"]
    fs = int(data["fs"])

if labels.tolist() != list(range(5)):
    raise ValueError(
        "Expected first five windows to contain all five classes."
    )

time = np.arange(reference.shape[1]) / fs

fig, axes = plt.subplots(
    5, 1, figsize=(14, 13), sharex=True
)

for index, ax in enumerate(axes):
    label = str(class_names[labels[index]])

    ax.plot(
        time,
        reference[index],
        color="#64748b",
        linewidth=1,
        alpha=0.75,
        label="Recorded reference ECG",
    )

    ax.plot(
        time,
        corrupted[index],
        color="#0284c7",
        linewidth=1,
        alpha=0.85,
        label="ECG with simulated artefact",
    )

    ax.set_title(label.replace("_", " ").title())
    ax.set_ylabel("Digital values")
    ax.grid(alpha=0.2)

    if index == 0:
        ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "TRACE — Controlled ECG Artefact Comparison",
    fontsize=16,
)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=160)
print("Saved:", OUTPUT)

for index in range(5):
    difference = corrupted[index] - reference[index]

    print(
        f"{class_names[index]}:"
        f" ECG difference RMS={np.sqrt(np.mean(difference**2)):.2f},"
        f" simulated IMU RMS={np.sqrt(np.mean(imu[index]**2)):.3f}"
    )

plt.show()
