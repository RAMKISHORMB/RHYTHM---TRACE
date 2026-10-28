
"""
TRACE — Stage 11.6
Continuous Motion and Repeated Artefact Stress Test

Evaluates transient-event detection under:
- Sustained simulated walking corruption
- Repeated simulated electrode pops
- Different corruption strengths
- Multiple random seeds

Reference events come from the same detector
running on uncorrupted MIT-BIH Record 100.

Research only. Not clinically validated.
"""

from itertools import product

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)


SCENARIOS = (
    "SUSTAINED_WALKING",
    "REPEATED_ELECTRODE_POPS",
)

STRENGTHS = (1.0, 2.0, 4.0)

SEEDS = (11, 42, 73)

START_SECONDS = 170.0
DURATION_SECONDS = 120.0

TARGET_EVENTS = (
    185.532,
    208.292,
    276.608,
)

MATCH_TOLERANCE_SECONDS = 0.15


def generate_corruption(
    scenario: str,
    strength: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """
    Generate a full-length ECG with corruption
    applied between 170 and 290 seconds.

    Corruption is additive and synthetic.
    """

    reference = load_evaluation_ecg()
    observed = reference.copy()

    start = int(round(START_SECONDS * FS))
    end = int(
        round(
            (START_SECONDS + DURATION_SECONDS) * FS
        )
    )

    if start < 0 or end > len(observed):
        raise ValueError(
            "Corruption interval exceeds ECG length."
        )

    rng = np.random.default_rng(seed)

    t = np.arange(end - start) / FS

    if scenario == "SUSTAINED_WALKING":

        artefact = (
            0.18 * np.sin(2 * np.pi * 1.8 * t)
            + 0.08 * np.sin(2 * np.pi * 0.4 * t)
            + rng.normal(
                0.0,
                0.035,
                size=len(t),
            )
        )

    elif scenario == "REPEATED_ELECTRODE_POPS":

        artefact = rng.normal(
            0.0,
            0.008,
            size=len(t),
        )

        # Generate a synthetic pop every 5 seconds,
        # with a reproducible small timing offset.
        pop_times = np.arange(
            2.0,
            DURATION_SECONDS,
            5.0,
        )

        for pop_time in pop_times:

            offset = rng.uniform(-0.3, 0.3)

            center = pop_time + offset

            amplitude = rng.choice(
                [-1.5, 1.5]
            )

            artefact += amplitude * np.exp(
                -0.5
                * ((t - center) / 0.035) ** 2
            )

    else:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    scaled_artefact = strength * artefact

    observed[start:end] += scaled_artefact

    rms = float(
        np.sqrt(
            np.mean(scaled_artefact ** 2)
        )
    )

    return observed, rms


def run_continuous_stress_test() -> list[dict]:
    """
    Run 18 controlled experiments.

    Detector failures are recorded as results.
    """

    reference = load_evaluation_ecg()

    reference_events = detect_events(reference)

    if not reference_events:
        raise RuntimeError(
            "No reference events detected."
        )

    results = []

    for experiment_id, (
        scenario,
        strength,
        seed,
    ) in enumerate(
        product(
            SCENARIOS,
            STRENGTHS,
            SEEDS,
        ),
        start=1,
    ):

        observed, corruption_rms = (
            generate_corruption(
                scenario=scenario,
                strength=strength,
                seed=seed,
            )
        )

        observed_events = detect_events(
            observed
        )

        comparison = match_events(
            reference_events,
            observed_events,
        )

        target_results = {}

        for target_time in TARGET_EVENTS:

            detected = any(
                abs(
                    match[
                        "reference_timestamp_seconds"
                    ]
                    - target_time
                ) <= MATCH_TOLERANCE_SECONDS
                for match in comparison["matches"]
            )

            target_results[
                str(target_time)
            ] = detected

        missed_targets = sum(
            not detected
            for detected in target_results.values()
        )

        results.append({
            "experiment_id": experiment_id,
            "scenario": scenario,
            "strength": strength,
            "seed": seed,
            "corruption_start_seconds": (
                START_SECONDS
            ),
            "corruption_duration_seconds": (
                DURATION_SECONDS
            ),
            "corruption_rms": round(
                corruption_rms, 6
            ),
            "reference_event_count": len(
                reference_events
            ),
            "observed_event_count": len(
                observed_events
            ),
            "matched_events": comparison[
                "matched_events"
            ],
            "missed_reference_events": comparison[
                "unmatched_reference_events"
            ],
            "extra_observed_events": comparison[
                "unmatched_observed_events"
            ],
            "agreement_f1": comparison[
                "agreement_f1"
            ],
            "missed_targets": missed_targets,
            "target_185_532_detected": (
                target_results["185.532"]
            ),
            "target_208_292_detected": (
                target_results["208.292"]
            ),
            "target_276_608_detected": (
                target_results["276.608"]
            ),
            "motion_source": (
                "synthetic_corruption_model"
            ),
            "clinical_validated": False,
        })

    return results
