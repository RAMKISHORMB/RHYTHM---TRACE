
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

with np.load(ROOT / "data/processed/100_250hz.npz") as data:
    ecg = data["ecg"][:, 0]
    fs = int(data["fs"])

with np.load(
    ROOT / "data/processed/100_annotations_250hz.npz"
) as data:
    beat_samples = data["samples"]
    symbols = data["symbols"]

# Plot the first 10 seconds of recorded ECG.
start = 0
end = min(10 * fs, len(ecg))

time = np.arange(start, end) / fs
signal = ecg[start:end]

mask = (beat_samples >= start) & (beat_samples < end)
visible_beats = beat_samples[mask]
visible_symbols = symbols[mask]

plt.figure(figsize=(15, 5))
plt.plot(time, signal, linewidth=1, label="Recorded ECG (MLII)")

for sample, symbol in zip(visible_beats, visible_symbols):
    beat_time = sample / fs
    plt.axvline(beat_time, color="red", alpha=0.65)
    plt.text(
        beat_time,
        np.max(signal),
        str(symbol),
        color="red",
        ha="center",
        fontsize=9,
    )

plt.title("TRACE — MIT-BIH Record 100: ECG and Beat Alignment")
plt.xlabel("Time (seconds)")
plt.ylabel("Uncalibrated digital values")
plt.grid(alpha=0.25)
plt.tight_layout()

output = ROOT / "data/processed/ecg_alignment.png"
plt.savefig(output, dpi=160)

print("Alignment plot saved:", output)
print("Visible beat annotations:", len(visible_beats))
print("Beat positions:", visible_beats.tolist())

plt.show()
