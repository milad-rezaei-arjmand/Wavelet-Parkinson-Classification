from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from matio import load_from_mat
from tqdm import tqdm


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FILE_INVENTORY_PATH = OUTPUT_DIR / "file_inventory.csv"

FS = 100
WINDOW_SIZE = 512
OVERLAP = 0.50
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

EXPECTED_CHANNELS = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]


# ============================================================
# Helper functions
# ============================================================

def get_activity_names(mat_data: dict[str, Any]) -> list[str]:
    """
    Find activities that have both left- and right-wrist tables.
    """

    left_activities = {
        name.removesuffix("_LeftWrist")
        for name, value in mat_data.items()
        if name.endswith("_LeftWrist")
        and isinstance(value, pd.DataFrame)
    }

    right_activities = {
        name.removesuffix("_RightWrist")
        for name, value in mat_data.items()
        if name.endswith("_RightWrist")
        and isinstance(value, pd.DataFrame)
    }

    return sorted(left_activities.intersection(right_activities))


def validate_table(
    table: pd.DataFrame,
    table_name: str,
) -> None:
    """Validate expected signal columns."""

    missing_channels = [
        channel
        for channel in EXPECTED_CHANNELS
        if channel not in table.columns
    ]

    if missing_channels:
        raise ValueError(
            f"{table_name} is missing channels: {missing_channels}"
        )


def get_window_starts(
    signal_length: int,
) -> list[int]:
    """Return all valid window start indices."""

    if signal_length < WINDOW_SIZE:
        return []

    return list(
        range(
            0,
            signal_length - WINDOW_SIZE + 1,
            STEP_SIZE,
        )
    )


# ============================================================
# Process one subject
# ============================================================

def process_subject(
    file_path: Path,
    subject_id: str,
    label: str,
) -> list[dict[str, Any]]:

    mat_data = load_from_mat(
        file_path,
        raw_data=False,
        add_table_attrs=True,
    )

    activities = get_activity_names(mat_data)

    rows: list[dict[str, Any]] = []

    for activity in activities:

        left_name = f"{activity}_LeftWrist"
        right_name = f"{activity}_RightWrist"

        left_table = mat_data[left_name]
        right_table = mat_data[right_name]

        validate_table(left_table, left_name)
        validate_table(right_table, right_name)

        left_length = len(left_table)
        right_length = len(right_table)

        usable_length = min(left_length, right_length)

        window_starts = get_window_starts(usable_length)

        for window_id, start_index in enumerate(window_starts):

            end_index = start_index + WINDOW_SIZE

            rows.append(
                {
                    "subject_id": subject_id,
                    "file_name": file_path.name,
                    "label": label,
                    "activity": activity,
                    "window_id": window_id,
                    "start_index": start_index,
                    "end_index": end_index,
                    "window_size": WINDOW_SIZE,
                    "step_size": STEP_SIZE,
                    "duration_seconds": WINDOW_SIZE / FS,
                    "left_signal_length": left_length,
                    "right_signal_length": right_length,
                    "usable_signal_length": usable_length,
                }
            )

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not FILE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {FILE_INVENTORY_PATH}"
        )

    file_inventory = pd.read_csv(FILE_INVENTORY_PATH)

    included_files = file_inventory[
        file_inventory["file_status"] == "included"
    ].copy()

    if included_files.empty:
        raise RuntimeError("No included files found.")

    all_rows: list[dict[str, Any]] = []

    errors: list[dict[str, str]] = []

    for row in tqdm(
        included_files.itertuples(index=False),
        total=len(included_files),
        desc="Building windows",
        unit="subject",
    ):
        file_path = DATA_DIR / row.file_name

        try:
            subject_rows = process_subject(
                file_path=file_path,
                subject_id=str(row.subject_id),
                label=str(row.standard_label),
            )

            all_rows.extend(subject_rows)

        except Exception as error:
            errors.append(
                {
                    "subject_id": str(row.subject_id),
                    "file_name": str(row.file_name),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    window_inventory = pd.DataFrame(all_rows)
    error_table = pd.DataFrame(errors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_inventory.to_csv(
        OUTPUT_DIR / "window_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    error_table.to_csv(
        OUTPUT_DIR / "window_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("WINDOW INVENTORY SUMMARY")
    print("=" * 70)

    print(f"\nIncluded subjects: {len(included_files)}")
    print(f"Created paired windows: {len(window_inventory)}")
    print(f"Processing errors: {len(error_table)}")

    print("\nWindows per class:")
    print(
        window_inventory["label"]
        .value_counts()
        .to_string()
    )

    print("\nWindows per activity:")
    print(
        window_inventory["activity"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nWindows per subject:")
    print(
        window_inventory.groupby("subject_id")
        .size()
        .describe()
        .to_string()
    )

    print("\nSignal lengths and number of windows:")
    print(
        window_inventory.groupby(
            "usable_signal_length"
        )["window_id"]
        .count()
        .to_string()
    )

    print("\nCreated files:")
    print(OUTPUT_DIR / "window_inventory.csv")
    print(OUTPUT_DIR / "window_errors.csv")


if __name__ == "__main__":
    main()