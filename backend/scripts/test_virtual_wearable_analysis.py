
"""
TRACE Stage 11.2 integration test.
"""

import numpy as np

from app.virtual_wearable_analysis import (
    analyze_virtual_wearable,
)


def main():
    for scenario in [
        "REST",
        "WALKING",
        "ELECTRODE_POP",
    ]:
        result = analyze_virtual_wearable(
            scenario=scenario,
            start_seconds=180.0,
            seed=42,
        )

        assert result["sample_count"] == 1000
        assert result["sampling_rate"] == 250
        assert np.isfinite(result["motion_rms"])
        assert np.isfinite(
            result["reference_corruption_rms"]
        )

        assert result["artifact_prediction"]
        assert isinstance(result["quality_flags"], list)
        assert result["clinical_validated"] is False

        print()
        print("Scenario:", result["scenario"])
        print(
            "Motion RMS:",
            round(result["motion_rms"], 4),
        )
        print(
            "Reference corruption RMS:",
            round(
                result["reference_corruption_rms"], 4
            ),
        )
        print(
            "Artefact prediction:",
            result["artifact_prediction"],
        )
        print(
            "Quality flags:",
            result["quality_flags"],
        )

    print()
    print(
        "PASS: Virtual wearable classifier "
        "and ECG quality integration."
    )


if __name__ == "__main__":
    main()
