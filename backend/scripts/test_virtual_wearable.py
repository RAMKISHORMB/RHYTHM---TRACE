
"""
TRACE Stage 11.1 verification.
"""

import numpy as np

from app.virtual_wearable import (
    generate_wearable_window,
)


def main():
    scenarios = [
        "REST",
        "WALKING",
        "ELECTRODE_POP",
    ]

    for scenario in scenarios:
        window = generate_wearable_window(
            scenario=scenario,
            start_seconds=180.0,
            seed=42,
        )

        reference = window.ecg_reference
        observed = window.ecg_observed
        motion = window.imu_simulated

        assert len(reference) == 1000
        assert len(observed) == 1000
        assert len(motion) == 1000

        assert window.fs == 250
        assert window.end_seconds == 184.0

        assert np.isfinite(reference).all()
        assert np.isfinite(observed).all()
        assert np.isfinite(motion).all()

        # The original recorded ECG is preserved.
        second_run = generate_wearable_window(
            scenario=scenario,
            start_seconds=180.0,
            seed=42,
        )

        assert np.array_equal(
            observed,
            second_run.ecg_observed,
        )

        assert np.array_equal(
            motion,
            second_run.imu_simulated,
        )

        ecg_difference = float(
            np.sqrt(
                np.mean((observed - reference) ** 2)
            )
        )

        motion_rms = float(
            np.sqrt(np.mean(motion ** 2))
        )

        if scenario == "REST":
            assert ecg_difference == 0.0
        else:
            assert ecg_difference > 0.0

        print(
            f"{scenario}: "
            f"samples={len(observed)}, "
            f"motion_rms={motion_rms:.4f}, "
            f"ecg_difference={ecg_difference:.4f}"
        )

    print(
        "PASS: All virtual wearable scenarios verified."
    )


if __name__ == "__main__":
    main()
