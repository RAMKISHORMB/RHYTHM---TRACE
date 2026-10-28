
"""
TRACE — Stage 11.6 Benchmark Runner

Runs 18 continuous artefact experiments
and exports the results to CSV.
"""

import csv
from pathlib import Path

from app.continuous_motion_stress import (
    SCENARIOS,
    STRENGTHS,
    SEEDS,
    run_continuous_stress_test,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "benchmarks"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "trace_continuous_motion_stress.csv"
)


def main():

    print("=" * 78)
    print("TRACE — STAGE 11.6")
    print("CONTINUOUS MOTION STRESS TEST")
    print("=" * 78)

    results = run_continuous_stress_test()

    expected = (
        len(SCENARIOS)
        * len(STRENGTHS)
        * len(SEEDS)
    )

    assert len(results) == expected == 18

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"{'ID':<4}"
        f"{'Scenario':<26}"
        f"{'Power':<8}"
        f"{'Seed':<7}"
        f"{'Miss':<7}"
        f"{'Extra':<8}"
        f"{'F1':<7}"
    )

    print("-" * 78)

    for result in results:

        print(
            f"{result['experiment_id']:<4}"
            f"{result['scenario']:<26}"
            f"{result['strength']:<8}"
            f"{result['seed']:<7}"
            f"{result['missed_reference_events']:<7}"
            f"{result['extra_observed_events']:<8}"
            f"{result['agreement_f1']:<7}"
        )

    experiments_with_misses = sum(
        result["missed_reference_events"] > 0
        for result in results
    )

    experiments_with_extra_events = sum(
        result["extra_observed_events"] > 0
        for result in results
    )

    mean_f1 = sum(
        result["agreement_f1"]
        for result in results
    ) / len(results)

    print()
    print("=" * 78)

    print(
        "Total experiments:",
        len(results),
    )

    print(
        "Experiments with missed events:",
        experiments_with_misses,
    )

    print(
        "Experiments with extra events:",
        experiments_with_extra_events,
    )

    print(
        "Mean agreement F1:",
        round(mean_f1, 4),
    )

    print(
        "CSV saved:",
        OUTPUT_FILE,
    )

    print()
    print(
        "PASS: All 18 continuous-motion "
        "experiments executed."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
