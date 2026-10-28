
"""
TRACE — Stage 12.3
Register the unified analysis router.

Makes an additive change to app/main.py.
Creates a backup before editing.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MAIN = ROOT / "app" / "main.py"

BACKUP = ROOT / "app" / "main_stage12_3_backup.txt"

IMPORT_LINE = (
    "from app.trace_analysis_api import "
    "router as trace_analysis_router"
)

INCLUDE_LINE = (
    "app.include_router(trace_analysis_router)"
)


def main():
    if not MAIN.exists():
        raise FileNotFoundError(MAIN)

    source = MAIN.read_text(
        encoding="utf-8"
    )

    if (
        IMPORT_LINE in source
        and INCLUDE_LINE in source
    ):
        print(
            "TRACE router already registered."
        )
        return

    if "app = FastAPI(" not in source:
        raise RuntimeError(
            "Cannot locate app = FastAPI(...). "
            "No changes made."
        )

    if (
        IMPORT_LINE in source
        or INCLUDE_LINE in source
    ):
        raise RuntimeError(
            "Partial registration detected. "
            "Inspect app/main.py manually."
        )

    # Keep the import at module level and register
    # the router at the end of the existing file.
    #
    # This assumes the application object is named
    # 'app' and exists at module scope.
    updated = (
        IMPORT_LINE
        + "\n"
        + source.rstrip()
        + "\n\n"
        + INCLUDE_LINE
        + "\n"
    )

    BACKUP.write_text(
        source,
        encoding="utf-8",
    )

    MAIN.write_text(
        updated,
        encoding="utf-8",
    )

    print("Backup:", BACKUP)
    print("Updated:", MAIN)
    print("TRACE router registered successfully.")


if __name__ == "__main__":
    main()
