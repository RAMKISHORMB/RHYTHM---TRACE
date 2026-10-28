
"""
TRACE — Stage 11.5 Benchmark Runner

Runs 24 stress experiments and exports CSV.
"""

import csv
from pathlib import Path

from app.event_stress_test import (
    SCENARIOS,
    STRENGTHS,
    EVENT_TIMES,
    run_stress_test,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "benchmarks"

OUTPUT_FILE = (
    OUTPUT_DIR / "trace_event_stress_test.csv"
)


def main():

    print("=" * 72)
    print("TRACE — STAGE 11.5")
    print("TRANSIENT EVENT STRESS TEST")
    print("=" * 72)

    results = run_stress_test()

    expected_count = (
        len(SCENARIOS)
        * len(STRENGTHS)
        * len(EVENT_TIMES)
    )

    assert len(results) == expected_count == 24

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
        f"{'Scenario':<16}"
        f"{'Power':<8}"
        f"{'Target':<12}"
        f"{'Found':<9}"
        f"{'Miss':<7}"
        f"{'Extra':<7}"
        f"{'F1':<7}"
    )

    print("-" * 72)

    for result in results:

        print(
            f"{result['experiment_id']:<4}"
            f"{result['scenario']:<16}"
            f"{result['strength']:<8}"
            f"{result['target_event_seconds']:<12}"
            f"{str(result['target_event_detected']):<9}"
            f"{result['missed_reference_events']:<7}"
            f"{result['extra_observed_events']:<7}"
            f"{result['agreement_f1']:<7}"
        )

    missed_target = sum(
        not result["target_event_detected"]
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
    print("=" * 72)

    print("Total experiments:", len(results))
    print("Missed target cases:", missed_target)

    print(
        "Experiments with extra events:",
        experiments_with_extra_events,
    )

    print(
        "Mean agreement F1:",
        round(mean_f1, 4),
    )

    print("CSV saved:", OUTPUT_FILE)

    print()
    print(
        "PASS: All 24 stress-test experiments executed."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
