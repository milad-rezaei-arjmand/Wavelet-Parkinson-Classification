from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pywt

from matio import load_from_mat
from scipy.stats import kurtosis, skew
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

WINDOW_INVENTORY_PATH = OUTPUT_DIR / "window_inventory.csv"

FEATURE_OUTPUT_PATH = (
    OUTPUT_DIR / "dwt_features_db4_level5.parquet"
)

FEATURE_MANIFEST_PATH = (
    OUTPUT_DIR / "dwt_feature_manifest.csv"
)

ERROR_OUTPUT_PATH = (
    OUTPUT_DIR / "dwt_extraction_errors.csv"
)


# ============================================================
# DWT settings
# ============================================================

WAVELET_NAME = "db4"
DECOMPOSITION_LEVEL = 5
DWT_MODE = "symmetric"

EXPECTED_WINDOW_SIZE = 512

WRISTS = [
    "LeftWrist",
    "RightWrist",
]

CHANNELS = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]

BANDS = [
    "A5",
    "D5",
    "D4",
    "D3",
    "D2",
    "D1",
]

STATISTICS = [
    "mean",
    "variance",
    "std",
    "energy",
    "relative_energy",
    "entropy",
    "iqr",
    "skewness",
    "kurtosis",
]

META_COLUMNS = [
    "subject_id",
    "file_name",
    "label",
    "activity",
    "window_id",
    "start_index",
    "end_index",
    "window_size",
]

EPSILON = 1e-12


# ============================================================
# Feature-column generation
# ============================================================

def create_feature_name(
    wrist: str,
    channel: str,
    band: str,
    statistic: str,
) -> str:
    return (
        f"{wrist}__"
        f"{channel}__"
        f"{band}__"
        f"{statistic}"
    )


def create_feature_columns() -> list[str]:
    feature_columns: list[str] = []

    for wrist in WRISTS:
        for channel in CHANNELS:
            for band in BANDS:
                for statistic in STATISTICS:
                    feature_columns.append(
                        create_feature_name(
                            wrist=wrist,
                            channel=channel,
                            band=band,
                            statistic=statistic,
                        )
                    )

    return feature_columns


FEATURE_COLUMNS = create_feature_columns()

ALL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS


# ============================================================
# Safe statistical functions
# ============================================================

def safe_skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0

    if np.isclose(np.std(values), 0.0):
        return 0.0

    result = float(
        skew(
            values,
            bias=False,
            nan_policy="omit",
        )
    )

    if not np.isfinite(result):
        return 0.0

    return result


def safe_kurtosis(values: np.ndarray) -> float:
    """
    Pearson kurtosis:
        Normal distribution has kurtosis approximately 3.
    """

    if values.size < 4:
        return 0.0

    if np.isclose(np.std(values), 0.0):
        return 0.0

    result = float(
        kurtosis(
            values,
            fisher=False,
            bias=False,
            nan_policy="omit",
        )
    )

    if not np.isfinite(result):
        return 0.0

    return result


def calculate_entropy(
    coefficients: np.ndarray,
    energy: float,
) -> float:
    """
    Shannon wavelet entropy based on normalized
    squared coefficients.
    """

    if energy <= EPSILON:
        return 0.0

    power = np.square(coefficients)

    probabilities = power / energy

    probabilities = probabilities[
        probabilities > EPSILON
    ]

    if probabilities.size == 0:
        return 0.0

    entropy_value = -np.sum(
        probabilities * np.log2(probabilities)
    )

    if not np.isfinite(entropy_value):
        return 0.0

    return float(entropy_value)


# ============================================================
# DWT feature extraction
# ============================================================

def decompose_signal(
    signal: np.ndarray,
) -> list[np.ndarray]:

    # PyWavelets به ورودی writable و contiguous نیاز دارد
    signal = np.array(
        signal,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    coefficients = pywt.wavedec(
        data=signal,
        wavelet=WAVELET_NAME,
        mode=DWT_MODE,
        level=DECOMPOSITION_LEVEL,
    )

    if len(coefficients) != len(BANDS):
        raise RuntimeError(
            "Unexpected number of wavelet coefficient arrays: "
            f"{len(coefficients)}"
        )

    return [
        np.array(
            coefficient,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        for coefficient in coefficients
    ]

def extract_band_statistics(
    coefficients: np.ndarray,
    energy: float,
    relative_energy: float,
) -> dict[str, float]:

    mean_value = float(np.mean(coefficients))
    variance_value = float(np.var(coefficients))
    std_value = float(np.std(coefficients))

    q25, q75 = np.percentile(
        coefficients,
        [25, 75],
    )

    iqr_value = float(q75 - q25)

    return {
        "mean": mean_value,
        "variance": variance_value,
        "std": std_value,
        "energy": energy,
        "relative_energy": relative_energy,
        "entropy": calculate_entropy(
            coefficients=coefficients,
            energy=energy,
        ),
        "iqr": iqr_value,
        "skewness": safe_skewness(coefficients),
        "kurtosis": safe_kurtosis(coefficients),
    }


def extract_channel_features(
    signal: np.ndarray,
    wrist: str,
    channel: str,
) -> dict[str, float]:

    coefficients = decompose_signal(signal)

    energies = [
        float(np.sum(np.square(band_coefficients)))
        for band_coefficients in coefficients
    ]

    total_energy = float(np.sum(energies))

    channel_features: dict[str, float] = {}

    for band_name, band_coefficients, energy in zip(
        BANDS,
        coefficients,
        energies,
    ):
        if total_energy <= EPSILON:
            relative_energy = 0.0
        else:
            relative_energy = energy / total_energy

        statistics = extract_band_statistics(
            coefficients=band_coefficients,
            energy=energy,
            relative_energy=relative_energy,
        )

        for statistic_name, statistic_value in statistics.items():
            feature_name = create_feature_name(
                wrist=wrist,
                channel=channel,
                band=band_name,
                statistic=statistic_name,
            )

            channel_features[feature_name] = statistic_value

    return channel_features


def extract_wrist_features(
    window_data: np.ndarray,
    wrist: str,
) -> dict[str, float]:

    if window_data.shape != (
        EXPECTED_WINDOW_SIZE,
        len(CHANNELS),
    ):
        raise ValueError(
            f"Unexpected window shape for {wrist}: "
            f"{window_data.shape}"
        )

    wrist_features: dict[str, float] = {}

    for channel_index, channel_name in enumerate(CHANNELS):
        signal = window_data[:, channel_index]

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                f"Non-finite values in {wrist}/{channel_name}"
            )

        channel_features = extract_channel_features(
            signal=signal,
            wrist=wrist,
            channel=channel_name,
        )

        wrist_features.update(channel_features)

    return wrist_features


def extract_paired_window_features(
    left_window: np.ndarray,
    right_window: np.ndarray,
) -> dict[str, float]:

    features: dict[str, float] = {}

    features.update(
        extract_wrist_features(
            window_data=left_window,
            wrist="LeftWrist",
        )
    )

    features.update(
        extract_wrist_features(
            window_data=right_window,
            wrist="RightWrist",
        )
    )

    if len(features) != len(FEATURE_COLUMNS):
        raise RuntimeError(
            "Unexpected feature count: "
            f"{len(features)} instead of "
            f"{len(FEATURE_COLUMNS)}"
        )

    return features


# ============================================================
# Validation
# ============================================================

def validate_signal_table(
    table: pd.DataFrame,
    table_name: str,
) -> None:

    missing_channels = [
        channel
        for channel in CHANNELS
        if channel not in table.columns
    ]

    if missing_channels:
        raise ValueError(
            f"{table_name} missing channels: "
            f"{missing_channels}"
        )


def table_to_array(
    table: pd.DataFrame,
) -> np.ndarray:

    numeric_table = table[CHANNELS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # ساخت آرایه مستقل، پیوسته و قابل‌نوشتن
    array = np.array(
        numeric_table.to_numpy(dtype=np.float64),
        dtype=np.float64,
        order="C",
        copy=True,
    )

    if not array.flags.writeable:
        array = array.copy(order="C")

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Signal table contains NaN or Inf values."
        )

    return array

# ============================================================
# Feature manifest
# ============================================================

def create_feature_manifest() -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    descriptions = {
        "mean": "Mean of wavelet coefficients",
        "variance": "Variance of wavelet coefficients",
        "std": "Standard deviation of wavelet coefficients",
        "energy": "Sum of squared wavelet coefficients",
        "relative_energy": (
            "Band energy divided by total channel energy"
        ),
        "entropy": (
            "Shannon entropy of normalized squared coefficients"
        ),
        "iqr": (
            "75th percentile minus 25th percentile"
        ),
        "skewness": (
            "Bias-corrected coefficient skewness"
        ),
        "kurtosis": (
            "Bias-corrected Pearson kurtosis"
        ),
    }

    for wrist in WRISTS:
        for channel in CHANNELS:
            for band in BANDS:
                for statistic in STATISTICS:
                    rows.append(
                        {
                            "feature_name": create_feature_name(
                                wrist=wrist,
                                channel=channel,
                                band=band,
                                statistic=statistic,
                            ),
                            "wrist": wrist,
                            "channel": channel,
                            "band": band,
                            "statistic": statistic,
                            "description": descriptions[statistic],
                            "wavelet": WAVELET_NAME,
                            "decomposition_level": (
                                DECOMPOSITION_LEVEL
                            ),
                            "boundary_mode": DWT_MODE,
                        }
                    )

    return pd.DataFrame(rows)


# ============================================================
# Batch processing
# ============================================================

def prepare_batch_dataframe(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:

    batch = pd.DataFrame(
        rows,
        columns=ALL_COLUMNS,
    )

    text_columns = [
        "subject_id",
        "file_name",
        "label",
        "activity",
    ]

    integer_columns = [
        "window_id",
        "start_index",
        "end_index",
        "window_size",
    ]

    batch[text_columns] = batch[text_columns].astype(str)

    batch[integer_columns] = batch[integer_columns].astype(
        np.int32
    )

    batch[FEATURE_COLUMNS] = batch[
        FEATURE_COLUMNS
    ].astype(np.float32)

    feature_array = batch[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if not np.all(np.isfinite(feature_array)):
        invalid_count = int(
            np.size(feature_array)
            - np.isfinite(feature_array).sum()
        )

        raise ValueError(
            f"Feature batch contains "
            f"{invalid_count} non-finite values."
        )

    return batch


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not WINDOW_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Window inventory not found: "
            f"{WINDOW_INVENTORY_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_inventory = pd.read_csv(
        WINDOW_INVENTORY_PATH,
        dtype={
            "subject_id": "string",
            "file_name": "string",
            "label": "string",
            "activity": "string",
        },
    )

    required_columns = {
        "file_name",
        "label",
        "activity",
        "window_id",
        "start_index",
        "end_index",
        "window_size",
    }

    missing_inventory_columns = (
        required_columns
        - set(window_inventory.columns)
    )

    if missing_inventory_columns:
        raise ValueError(
            "Window inventory is missing columns: "
            f"{sorted(missing_inventory_columns)}"
        )

    if len(window_inventory) != 15975:
        print(
            "Warning: expected 15975 windows, "
            f"but found {len(window_inventory)}."
        )

    if FEATURE_OUTPUT_PATH.exists():
        FEATURE_OUTPUT_PATH.unlink()

    feature_manifest = create_feature_manifest()

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    grouped_files = window_inventory.groupby(
        "file_name",
        sort=True,
    )

    parquet_writer: pq.ParquetWriter | None = None
    parquet_schema: pa.Schema | None = None

    processed_rows = 0
    processed_subjects = 0

    class_counter: Counter[str] = Counter()
    activity_counter: Counter[str] = Counter()

    extraction_errors: list[dict[str, str]] = []

    try:
        for file_name, subject_windows in tqdm(
            grouped_files,
            total=window_inventory["file_name"].nunique(),
            desc="Extracting DWT features",
            unit="subject",
        ):
            file_name = str(file_name)
            file_path = DATA_DIR / file_name

            subject_id = Path(file_name).stem

            try:
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"MAT file not found: {file_path}"
                    )

                mat_data = load_from_mat(
                    file_path,
                    raw_data=False,
                    add_table_attrs=True,
                )

                batch_rows: list[dict[str, Any]] = []

                for window_row in subject_windows.itertuples(
                    index=False
                ):
                    activity = str(window_row.activity)
                    label = str(window_row.label)

                    left_name = (
                        f"{activity}_LeftWrist"
                    )

                    right_name = (
                        f"{activity}_RightWrist"
                    )

                    if left_name not in mat_data:
                        raise KeyError(
                            f"Missing variable: {left_name}"
                        )

                    if right_name not in mat_data:
                        raise KeyError(
                            f"Missing variable: {right_name}"
                        )

                    left_table = mat_data[left_name]
                    right_table = mat_data[right_name]

                    if not isinstance(
                        left_table,
                        pd.DataFrame,
                    ):
                        raise TypeError(
                            f"{left_name} is not a DataFrame."
                        )

                    if not isinstance(
                        right_table,
                        pd.DataFrame,
                    ):
                        raise TypeError(
                            f"{right_name} is not a DataFrame."
                        )

                    validate_signal_table(
                        left_table,
                        left_name,
                    )

                    validate_signal_table(
                        right_table,
                        right_name,
                    )

                    left_array = table_to_array(left_table)
                    right_array = table_to_array(right_table)

                    start_index = int(
                        window_row.start_index
                    )

                    end_index = int(
                        window_row.end_index
                    )

                    window_size = int(
                        window_row.window_size
                    )

                    if window_size != EXPECTED_WINDOW_SIZE:
                        raise ValueError(
                            "Unexpected window size: "
                            f"{window_size}"
                        )

                    if (
                        end_index - start_index
                        != EXPECTED_WINDOW_SIZE
                    ):
                        raise ValueError(
                            "Window index size mismatch."
                        )

                    if end_index > len(left_array):
                        raise IndexError(
                            f"Left window exceeds "
                            f"{left_name} length."
                        )

                    if end_index > len(right_array):
                        raise IndexError(
                            f"Right window exceeds "
                            f"{right_name} length."
                        )

                    left_window = left_array[
                        start_index:end_index,
                        :,
                    ]

                    right_window = right_array[
                        start_index:end_index,
                        :,
                    ]

                    features = (
                        extract_paired_window_features(
                            left_window=left_window,
                            right_window=right_window,
                        )
                    )

                    result_row: dict[str, Any] = {
                        "subject_id": subject_id,
                        "file_name": file_name,
                        "label": label,
                        "activity": activity,
                        "window_id": int(
                            window_row.window_id
                        ),
                        "start_index": start_index,
                        "end_index": end_index,
                        "window_size": window_size,
                    }

                    result_row.update(features)

                    batch_rows.append(result_row)

                    class_counter[label] += 1
                    activity_counter[activity] += 1

                batch_dataframe = prepare_batch_dataframe(
                    batch_rows
                )

                arrow_table = pa.Table.from_pandas(
                    batch_dataframe,
                    preserve_index=False,
                )

                if parquet_writer is None:
                    parquet_schema = arrow_table.schema

                    parquet_writer = pq.ParquetWriter(
                        where=str(FEATURE_OUTPUT_PATH),
                        schema=parquet_schema,
                        compression="snappy",
                    )

                elif arrow_table.schema != parquet_schema:
                    arrow_table = arrow_table.cast(
                        parquet_schema
                    )

                parquet_writer.write_table(arrow_table)

                processed_rows += len(batch_dataframe)
                processed_subjects += 1

            except Exception as error:
                extraction_errors.append(
                    {
                        "subject_id": subject_id,
                        "file_name": file_name,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    error_dataframe = pd.DataFrame(
        extraction_errors,
        columns=[
            "subject_id",
            "file_name",
            "error_type",
            "error_message",
        ],
    )

    error_dataframe.to_csv(
        ERROR_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("DWT FEATURE EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"\nWavelet: {WAVELET_NAME}")
    print(
        "Decomposition level: "
        f"{DECOMPOSITION_LEVEL}"
    )
    print(f"Boundary mode: {DWT_MODE}")

    print(
        f"\nProcessed subjects: {processed_subjects}"
    )
    print(f"Created feature rows: {processed_rows}")
    print(
        f"DWT features per row: "
        f"{len(FEATURE_COLUMNS)}"
    )
    print(
        f"Total columns: "
        f"{len(ALL_COLUMNS)}"
    )
    print(
        f"Extraction errors: "
        f"{len(error_dataframe)}"
    )

    print("\nRows per class:")

    for label, count in class_counter.items():
        print(f"{label}: {count}")

    print("\nRows per activity:")

    for activity in sorted(activity_counter):
        print(
            f"{activity}: "
            f"{activity_counter[activity]}"
        )

    print("\nCreated files:")
    print(FEATURE_OUTPUT_PATH)
    print(FEATURE_MANIFEST_PATH)
    print(ERROR_OUTPUT_PATH)

    if processed_rows != len(window_inventory):
        raise RuntimeError(
            "The number of extracted rows does not match "
            "the window inventory. "
            f"Expected {len(window_inventory)}, "
            f"created {processed_rows}."
        )

    if not error_dataframe.empty:
        raise RuntimeError(
            "DWT extraction completed with errors. "
            f"Inspect: {ERROR_OUTPUT_PATH}"
        )


if __name__ == "__main__":
    main()