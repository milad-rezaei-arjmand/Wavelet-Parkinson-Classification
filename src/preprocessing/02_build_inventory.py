from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matio import load_from_mat
from tqdm import tqdm


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FS = 100

EXPECTED_CHANNELS = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]


# ============================================================
# Label processing
# ============================================================

HEALTHY_LABELS = {
    "healthy",
    "control",
    "healthy control",
}

PARKINSON_LABELS = {
    "parkinson",
    "parkinsons",
    "parkinson's",
    "parkinson disease",
    "parkinson's disease",
    "patient",
    "parkinson patient",
    "parkinson/patient",
    "pd",
}


def convert_to_text(value: Any) -> str:
    """Convert MATLAB/NumPy label values to a clean string."""

    if value is None:
        return ""

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""

        value = value.squeeze()

        if np.asarray(value).size == 1:
            value = np.asarray(value).item()
        else:
            value = " ".join(map(str, np.asarray(value).ravel()))

    text = str(value)

    text = (
        text.replace("\x00", "")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .lower()
    )

    return " ".join(text.split())


def standardize_label(raw_label: Any) -> tuple[str | None, str]:
    """
    Convert raw labels to:
        Healthy
        Parkinson
        None for invalid/ambiguous labels
    """

    cleaned_label = convert_to_text(raw_label)

    if not cleaned_label:
        return None, "missing_label"

    if cleaned_label in HEALTHY_LABELS:
        return "Healthy", "valid"

    if cleaned_label in PARKINSON_LABELS:
        return "Parkinson", "valid"

    return None, f"invalid_or_ambiguous_label:{cleaned_label}"


# ============================================================
# Signal-name processing
# ============================================================

def parse_signal_name(
    variable_name: str,
) -> tuple[str | None, str | None]:
    """
    Example:
        CrossArms_LeftWrist
    becomes:
        activity = CrossArms
        wrist = LeftWrist
    """

    match = re.match(
        r"^(?P<activity>.+)_(?P<wrist>LeftWrist|RightWrist)$",
        variable_name,
    )

    if match is None:
        return None, None

    return match.group("activity"), match.group("wrist")


# ============================================================
# DataFrame inspection
# ============================================================

def inspect_signal_table(
    file_name: str,
    subject_id: str,
    label: str | None,
    variable_name: str,
    table: pd.DataFrame,
) -> dict[str, Any]:

    activity, wrist = parse_signal_name(variable_name)

    missing_channels = [
        channel
        for channel in EXPECTED_CHANNELS
        if channel not in table.columns
    ]

    extra_channels = [
        str(column)
        for column in table.columns
        if column not in EXPECTED_CHANNELS
    ]

    if missing_channels:
        numeric_array = np.empty((0, 0))
        nan_count = 0
        inf_count = 0
        constant_channel_count = 0
        signal_status = "invalid"
        signal_reason = (
            "missing_channels:" + "|".join(missing_channels)
        )

    else:
        numeric_data = table[EXPECTED_CHANNELS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        numeric_array = numeric_data.to_numpy(dtype=float)

        nan_count = int(np.isnan(numeric_array).sum())
        inf_count = int(np.isinf(numeric_array).sum())

        channel_std = np.nanstd(numeric_array, axis=0)
        constant_channel_count = int(
            np.sum(np.isclose(channel_std, 0))
        )

        reasons = []

        if activity is None or wrist is None:
            reasons.append("invalid_variable_name")

        if len(table) == 0:
            reasons.append("empty_table")

        if nan_count > 0:
            reasons.append(f"nan_values:{nan_count}")

        if inf_count > 0:
            reasons.append(f"inf_values:{inf_count}")

        if constant_channel_count > 0:
            reasons.append(
                f"constant_channels:{constant_channel_count}"
            )

        if reasons:
            signal_status = "warning"
            signal_reason = ";".join(reasons)
        else:
            signal_status = "valid"
            signal_reason = "valid"

    return {
        "subject_id": subject_id,
        "file_name": file_name,
        "label": label or "",
        "variable_name": variable_name,
        "activity": activity or "",
        "wrist": wrist or "",
        "n_samples": int(table.shape[0]),
        "n_columns": int(table.shape[1]),
        "duration_seconds": table.shape[0] / FS,
        "column_names": "|".join(map(str, table.columns)),
        "missing_channels": "|".join(missing_channels),
        "extra_channels": "|".join(extra_channels),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "constant_channel_count": constant_channel_count,
        "signal_status": signal_status,
        "signal_reason": signal_reason,
    }


# ============================================================
# MAT-file inspection
# ============================================================

def inspect_file(
    file_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    subject_id = file_path.stem
    file_name = file_path.name

    try:
        mat_data = load_from_mat(
            file_path,
            raw_data=False,
            add_table_attrs=True,
        )

    except Exception as error:
        file_result = {
            "subject_id": subject_id,
            "file_name": file_name,
            "raw_label": "",
            "standard_label": "",
            "n_signal_tables": 0,
            "n_valid_signals": 0,
            "n_warning_signals": 0,
            "file_status": "excluded",
            "file_reason": f"read_error:{type(error).__name__}:{error}",
        }

        return file_result, []

    raw_label = mat_data.get("Label")
    standard_label, label_reason = standardize_label(raw_label)

    signal_results: list[dict[str, Any]] = []

    for variable_name, variable_value in mat_data.items():

        if not isinstance(variable_value, pd.DataFrame):
            continue

        signal_results.append(
            inspect_signal_table(
                file_name=file_name,
                subject_id=subject_id,
                label=standard_label,
                variable_name=str(variable_name),
                table=variable_value,
            )
        )

    n_valid_signals = sum(
        row["signal_status"] == "valid"
        for row in signal_results
    )

    n_warning_signals = sum(
        row["signal_status"] == "warning"
        for row in signal_results
    )

    file_reasons = []

    if standard_label is None:
        file_reasons.append(label_reason)

    if len(signal_results) == 0:
        file_reasons.append("no_signal_tables")

    if n_valid_signals == 0:
        file_reasons.append("no_fully_valid_signals")

    if standard_label is not None and n_valid_signals > 0:
        file_status = "included"

        if n_warning_signals > 0:
            file_reason = (
                f"included_with_{n_warning_signals}_signal_warnings"
            )
        else:
            file_reason = "valid"
    else:
        file_status = "excluded"
        file_reason = ";".join(file_reasons)

    file_result = {
        "subject_id": subject_id,
        "file_name": file_name,
        "raw_label": convert_to_text(raw_label),
        "standard_label": standard_label or "",
        "n_signal_tables": len(signal_results),
        "n_valid_signals": n_valid_signals,
        "n_warning_signals": n_warning_signals,
        "file_status": file_status,
        "file_reason": file_reason,
    }

    return file_result, signal_results


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    mat_files = sorted(DATA_DIR.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(
            f"No MAT files found in: {DATA_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Number of MAT files: {len(mat_files)}")
    print("=" * 70)

    file_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    for file_path in tqdm(
        mat_files,
        desc="Inspecting files",
        unit="file",
    ):
        file_result, file_signals = inspect_file(file_path)

        file_rows.append(file_result)
        signal_rows.extend(file_signals)

    file_inventory = pd.DataFrame(file_rows)
    signal_inventory = pd.DataFrame(signal_rows)

    excluded_files = file_inventory[
        file_inventory["file_status"] == "excluded"
    ].copy()

    warning_signals = signal_inventory[
        signal_inventory["signal_status"] != "valid"
    ].copy()

    file_inventory.to_csv(
        OUTPUT_DIR / "file_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    signal_inventory.to_csv(
        OUTPUT_DIR / "signal_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded_files.to_csv(
        OUTPUT_DIR / "excluded_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    warning_signals.to_csv(
        OUTPUT_DIR / "warning_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("\nFile status:")
    print(
        file_inventory["file_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nClass distribution:")
    print(
        file_inventory["standard_label"]
        .replace("", "Invalid/Excluded")
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSignal tables per file:")
    print(
        file_inventory["n_signal_tables"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if not signal_inventory.empty:

        print("\nSignal lengths:")
        print(
            signal_inventory["n_samples"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nActivities:")
        print(
            signal_inventory["activity"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nWrists:")
        print(
            signal_inventory["wrist"]
            .value_counts()
            .to_string()
        )

        print("\nSignal status:")
        print(
            signal_inventory["signal_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nCreated output files:")

    for output_name in [
        "file_inventory.csv",
        "signal_inventory.csv",
        "excluded_files.csv",
        "warning_signals.csv",
    ]:
        print(OUTPUT_DIR / output_name)


if __name__ == "__main__":
    main()