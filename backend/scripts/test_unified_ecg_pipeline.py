
"""
TRACE — Stage 12.1
Unified ECG Pipeline Verification

Checks:
- REST
- WALKING
- ELECTRODE_POP
- Window structure
- Compensation output
- Review routing
- Non-diagnostic metadata
"""

import json

import numpy as np

from app.unified_ecg_pipeline import (
    analyze_unified_window,
)


SCENARIOS = (
    "REST",
    "WALKING",
    "ELECTRODE_POP",
)


def main():
    print("=" * 72)
    print("TRACE STAGE 12.1 — UNIFIED ECG PIPELINE")
    print("=" * 72)

    for scenario in SCENARIOS:
        result = analyze_unified_window(
            scenario=scenario,
            start_seconds=180.0,
            seed=42,
        )

        data = result.to_dict()

        assert data["scenario"] == scenario

        assert data["sample_rate_hz"] == 250

        assert data["sample_count"] == 1000

        assert data["duration_seconds"] == 4.0

        assert len(data["observed_ecg"]) == 1000

        assert len(data["output_ecg"]) == 1000

        assert len(data["simulated_motion"]) == 1000

        assert np.isfinite(
            data["output_ecg"]
        ).all()

        assert data["trust_status"] in {
            "REVIEW_CANDIDATE",
            "MOTION_AFFECTED_REVIEW",
            "COMPENSATED_REVIEW",
            "LOW_CONFIDENCE_REVIEW",
        }

        assert (
            data["metadata"]["clinical_validated"]
            is False
        )

        assert (
            data["metadata"]["event_detection_performed"]
            is False
        )

        # Confirm the result can be serialized
        # into an API-compatible JSON payload.
        json.dumps(data, allow_nan=False)

        print()
        print("-" * 72)
        print("Scenario:", scenario)
        print("Artefact label:", data["artifact_label"])
        print("Quality flags:", data["quality_flags"])
        print(
            "Compensation:",
            data["compensation"]["decision"],
        )
        print(
            "Compensation reason:",
            data["compensation"]["reason"],
        )
        print("Trust status:", data["trust_status"])
        print("Trust reasons:", data["trust_reasons"])
        print("Samples:", data["sample_count"])
        print("JSON serialization: PASS")

    print()
    print("=" * 72)
    print("PASS: All three unified window scenarios executed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
