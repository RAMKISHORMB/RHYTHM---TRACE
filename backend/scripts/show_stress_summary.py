
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT
    / "data"
    / "benchmarks"
    / "trace_continuous_motion_stress.csv"
)


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {CSV_FILE}"
        )

    with CSV_FILE.open(
        newline="",
        encoding="utf-8",
    ) as file:
        results = list(csv.DictReader(file))

    if not results:
        raise ValueError("Benchmark CSV is empty.")

    total = len(results)

    missed = sum(
        int(row["missed_reference_events"])
        for row in results
    )

    extra = sum(
        int(row["extra_observed_events"])
        for row in results
    )

    mean_f1 = sum(
        float(row["agreement_f1"])
        for row in results
    ) / total

    worst = sorted(
        results,
        key=lambda row: float(
            row["agreement_f1"]
        ),
    )[:5]

    print("=" * 60)
    print("TRACE STAGE 11.6 - BENCHMARK SUMMARY")
    print("=" * 60)

    print("Total experiments:", total)
    print("Total missed references:", missed)
    print("Total extra detections:", extra)
    print("Mean agreement F1:", round(mean_f1, 4))

    print()
    print("WORST FIVE EXPERIMENTS")

    for row in worst:
        print(
            f"{row['scenario']} | "
            f"strength={row['strength']} | "
            f"seed={row['seed']} | "
            f"missed={row['missed_reference_events']} | "
            f"extra={row['extra_observed_events']} | "
            f"F1={row['agreement_f1']}"
        )

    print()
    print("SUMMARY COMPLETE")


if __name__ == "__main__":
    main()
