
"""
TRACE — Stage 11.10
Motion-Correlation Calibration

Controlled, research-only experiment.

A motion-informed ridge regression model is
calibrated on one ECG interval and evaluated
on a separate interval.

No clean ECG is used to fit the correction.
Clean ECG is used only for benchmark evaluation.

Not clinically validated.
"""

import numpy as np

from app.corrupted_event_evaluation import (
    FS,
    load_evaluation_ecg,
    detect_events,
    match_events,
)


TRAIN_START = 20
TRAIN_END = 120
TEST_START = 170
TEST_END = 290

TAPS = 5
RIDGE = 100.0

# Limit the correction relative to the
# variation of the observed ECG.
MAX_CORRECTION_FRACTION = 0.30

# Require an improvement on a held-out
# calibration interval before accepting.
MIN_VALIDATION_IMPROVEMENT = 0.02


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def lagged_matrix(motion, taps=TAPS):
    motion = np.asarray(motion, dtype=np.float64)

    matrix = np.zeros(
        (len(motion), taps),
        dtype=np.float64,
    )

    for lag in range(taps):
        if lag == 0:
            matrix[:, lag] = motion
        else:
            matrix[lag:, lag] = motion[:-lag]

    return matrix


def generate_case(
    strength=1.0,
    seed=42,
    informative=True,
):
    reference = np.asarray(
        load_evaluation_ecg(),
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    n = len(reference)

    # Broadband synthetic motion process.
    source_motion = rng.normal(size=n)

    # Short smoothing creates temporally
    # correlated motion rather than pure white noise.
    kernel = np.ones(9) / 9.0
    source_motion = np.convolve(
        source_motion,
        kernel,
        mode="same",
    )

    source_motion /= max(
        np.std(source_motion),
        1e-12,
    )

    delayed = np.concatenate(
        (
            np.zeros(3),
            source_motion[:-3],
        )
    )

    artifact = strength * (
        6.0 * source_motion
        + 3.0 * delayed
    )

    observed = reference + artifact

    if informative:
        motion_reference = source_motion.copy()
    else:
        # Independent reference without the
        # same periodic waveform.
        unrelated_rng = np.random.default_rng(
            seed + 10000
        )

        motion_reference = unrelated_rng.normal(
            size=n
        )

        motion_reference = np.convolve(
            motion_reference,
            kernel,
            mode="same",
        )

        motion_reference /= max(
            np.std(motion_reference),
            1e-12,
        )

    return reference, observed, motion_reference


def calibrate_and_evaluate(
    strength=1.0,
    seed=42,
    informative=True,
):
    reference, observed, motion = generate_case(
        strength=strength,
        seed=seed,
        informative=informative,
    )

    # Center using the observed training
    # interval only, not the clean reference.
    train_start = int(TRAIN_START * FS)
    train_end = int(TRAIN_END * FS)

    test_start = int(TEST_START * FS)
    test_end = int(TEST_END * FS)

    baseline = float(
        np.median(
            observed[train_start:train_end]
        )
    )

    motion_mean = float(
        np.mean(
            motion[train_start:train_end]
        )
    )

    centered_ecg = observed - baseline
    centered_motion = motion - motion_mean

    X = lagged_matrix(centered_motion)

    # Separate training interval into fitting
    # and validation portions.
    split = train_start + int(
        0.75 * (train_end - train_start)
    )

    X_fit = X[train_start:split]
    y_fit = centered_ecg[train_start:split]

    X_validation = X[split:train_end]
    y_validation = centered_ecg[split:train_end]

    gram = X_fit.T @ X_fit

    coefficients = np.linalg.solve(
        gram + RIDGE * np.eye(TAPS),
        X_fit.T @ y_fit,
    )

    validation_prediction = (
        X_validation @ coefficients
    )

    validation_before = np.var(
        y_validation
    )

    validation_after = np.var(
        y_validation - validation_prediction
    )

    validation_improvement = (
        1.0
        - validation_after
        / max(validation_before, 1e-12)
    )

    predicted_artifact = X @ coefficients

    test_prediction = predicted_artifact[
        test_start:test_end
    ]

    test_observed = observed[
        test_start:test_end
    ]

    correction_fraction = (
        rms(test_prediction)
        / max(
            np.std(test_observed),
            1e-12,
        )
    )

    accepted = bool(
        validation_improvement
        >= MIN_VALIDATION_IMPROVEMENT
        and correction_fraction
        <= MAX_CORRECTION_FRACTION
    )

    output = observed.copy()

    if accepted:
        output[test_start:test_end] -= (
            test_prediction
        )

    reference_test = reference[
        test_start:test_end
    ]

    observed_test = observed[
        test_start:test_end
    ]

    output_test = output[
        test_start:test_end
    ]

    raw_error = rms(
        observed_test - reference_test
    )

    output_error = rms(
        output_test - reference_test
    )

    reduction_percent = (
        100.0 * (
            raw_error - output_error
        ) / raw_error
        if raw_error > 0
        else 0.0
    )

    reference_events = detect_events(
        reference_test
    )

    raw_events = detect_events(
        observed_test
    )

    output_events = detect_events(
        output_test
    )

    raw_match = match_events(
        reference_events,
        raw_events,
    )

    output_match = match_events(
        reference_events,
        output_events,
    )

    return {
        "reference_type": (
            "INFORMATIVE"
            if informative
            else "UNRELATED"
        ),
        "strength": strength,
        "seed": seed,
        "accepted": accepted,
        "decision": (
            "ACCEPT_CORRECTION"
            if accepted
            else "PRESERVE_ORIGINAL"
        ),
        "validation_improvement": float(
            validation_improvement
        ),
        "correction_fraction": float(
            correction_fraction
        ),
        "raw_error_rms": float(raw_error),
        "output_error_rms": float(output_error),
        "reduction_percent": float(
            reduction_percent
        ),
        "raw_agreement_f1": raw_match[
            "agreement_f1"
        ],
        "output_agreement_f1": output_match[
            "agreement_f1"
        ],
        "missed_events": output_match[
            "unmatched_reference_events"
        ],
        "extra_events": output_match[
            "unmatched_observed_events"
        ],
        "clinical_validated": False,
    }
