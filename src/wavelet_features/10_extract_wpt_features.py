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

WINDOW_INVENTORY_PATH = (
    OUTPUT_DIR / "window_inventory.csv"
)

FEATURE_OUTPUT_PATH = (
    OUTPUT_DIR / "wpt_features_db4_level4.parquet"
)

FEATURE_MANIFEST_PATH = (
    OUTPUT_DIR / "wpt_feature_manifest.csv"
)

ERROR_OUTPUT_PATH = (
    OUTPUT_DIR / "wpt_extraction_errors.csv"
)


# ============================================================
# WPT settings
# ============================================================

FS = 100

WAVELET_NAME = "db4"
WPT_LEVEL = 4
WPT_MODE = "symmetric"

EXPECTED_WINDOW_SIZE = 512
EXPECTED_TERMINAL_BANDS = 2 ** WPT_LEVEL

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
# Determine frequency-ordered terminal WPT nodes
# ============================================================

def get_terminal_node_paths() -> list[str]:
    prototype_signal = np.zeros(
        EXPECTED_WINDOW_SIZE,
        dtype=np.float64,
    )

    packet = pywt.WaveletPacket(
        data=prototype_signal,
        wavelet=WAVELET_NAME,
        mode=WPT_MODE,
        maxlevel=WPT_LEVEL,
    )

    nodes = packet.get_level(
        WPT_LEVEL,
        order="freq",
        decompose=True,
    )

    paths = [
        node.path
        for node in nodes
    ]

    if len(paths) != EXPECTED_TERMINAL_BANDS:
        raise RuntimeError(
            "Unexpected number of WPT terminal nodes: "
            f"{len(paths)} instead of "
            f"{EXPECTED_TERMINAL_BANDS}"
        )

    if len(set(paths)) != len(paths):
        raise RuntimeError(
            "Duplicate WPT terminal-node paths found."
        )

    return paths


TERMINAL_NODE_PATHS = get_terminal_node_paths()


# ============================================================
# Feature names
# ============================================================

def create_band_name(
    band_index: int,
    node_path: str,
) -> str:
    return (
        f"L{WPT_LEVEL}_"
        f"B{band_index:02d}_"
        f"{node_path}"
    )


BAND_NAMES = [
    create_band_name(
        band_index=band_index,
        node_path=node_path,
    )
    for band_index, node_path
    in enumerate(TERMINAL_NODE_PATHS)
]


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
    columns: list[str] = []

    for wrist in WRISTS:
        for channel in CHANNELS:
            for band in BAND_NAMES:
                for statistic in STATISTICS:
                    columns.append(
                        create_feature_name(
                            wrist=wrist,
                            channel=channel,
                            band=band,
                            statistic=statistic,
                        )
                    )

    return columns


FEATURE_COLUMNS = create_feature_columns()
ALL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS


# ============================================================
# Safe statistics
# ============================================================

def safe_skewness(
    values: np.ndarray,
) -> float:

    if values.size < 3:
        return 0.0

    if np.isclose(
        np.std(values),
        0.0,
    ):
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


def safe_kurtosis(
    values: np.ndarray,
) -> float:
    """
    Pearson kurtosis.

    A normal distribution has kurtosis approximately 3.
    """

    if values.size < 4:
        return 0.0

    if np.isclose(
        np.std(values),
        0.0,
    ):
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
        probabilities
        * np.log2(probabilities)
    )

    if not np.isfinite(entropy_value):
        return 0.0

    return float(entropy_value)


def extract_band_statistics(
    coefficients: np.ndarray,
    energy: float,
    relative_energy: float,
) -> dict[str, float]:

    q25, q75 = np.percentile(
        coefficients,
        [25, 75],
    )

    return {
        "mean": float(
            np.mean(coefficients)
        ),
        "variance": float(
            np.var(coefficients)
        ),
        "std": float(
            np.std(coefficients)
        ),
        "energy": float(
            energy
        ),
        "relative_energy": float(
            relative_energy
        ),
        "entropy": calculate_entropy(
            coefficients=coefficients,
            energy=energy,
        ),
        "iqr": float(
            q75 - q25
        ),
        "skewness": safe_skewness(
            coefficients
        ),
        "kurtosis": safe_kurtosis(
            coefficients
        ),
    }


# ============================================================
# Signal and table validation
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
            f"{table_name} is missing channels: "
            f"{missing_channels}"
        )


def table_to_array(
    table: pd.DataFrame,
) -> np.ndarray:

    numeric_table = table[
        CHANNELS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Independent, contiguous and writable array
    array = np.array(
        numeric_table.to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        order="C",
        copy=True,
    )

    if not array.flags.writeable:
        array = array.copy(order="C")

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Signal table contains NaN or Inf."
        )

    return array


# ============================================================
# WPT decomposition
# ============================================================

def decompose_signal_wpt(
    signal: np.ndarray,
) -> list[np.ndarray]:

    signal = np.array(
        signal,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    if signal.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional signal, "
            f"found shape {signal.shape}."
        )

    if signal.size != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            f"Expected signal length "
            f"{EXPECTED_WINDOW_SIZE}, "
            f"found {signal.size}."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "WPT input contains NaN or Inf."
        )

    packet = pywt.WaveletPacket(
        data=signal,
        wavelet=WAVELET_NAME,
        mode=WPT_MODE,
        maxlevel=WPT_LEVEL,
    )

    nodes = packet.get_level(
        WPT_LEVEL,
        order="freq",
        decompose=True,
    )

    current_paths = [
        node.path
        for node in nodes
    ]

    if current_paths != TERMINAL_NODE_PATHS:
        raise RuntimeError(
            "Unexpected WPT terminal-node ordering."
        )

    coefficients = [
        np.array(
            node.data,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        for node in nodes
    ]

    if len(coefficients) != EXPECTED_TERMINAL_BANDS:
        raise RuntimeError(
            "Unexpected number of WPT bands: "
            f"{len(coefficients)}"
        )

    return coefficients


# ============================================================
# Feature extraction
# ============================================================

def extract_channel_features(
    signal: np.ndarray,
    wrist: str,
    channel: str,
) -> dict[str, float]:

    coefficient_bands = (
        decompose_signal_wpt(signal)
    )

    energies = [
        float(
            np.sum(
                np.square(coefficients)
            )
        )
        for coefficients
        in coefficient_bands
    ]

    total_energy = float(
        np.sum(energies)
    )

    channel_features: dict[str, float] = {}

    for (
        band_name,
        coefficients,
        energy,
    ) in zip(
        BAND_NAMES,
        coefficient_bands,
        energies,
    ):

        if total_energy <= EPSILON:
            relative_energy = 0.0
        else:
            relative_energy = (
                energy / total_energy
            )

        statistics = (
            extract_band_statistics(
                coefficients=coefficients,
                energy=energy,
                relative_energy=relative_energy,
            )
        )

        for (
            statistic_name,
            statistic_value,
        ) in statistics.items():

            feature_name = create_feature_name(
                wrist=wrist,
                channel=channel,
                band=band_name,
                statistic=statistic_name,
            )

            channel_features[
                feature_name
            ] = statistic_value

    expected_channel_features = (
        EXPECTED_TERMINAL_BANDS
        * len(STATISTICS)
    )

    if (
        len(channel_features)
        != expected_channel_features
    ):
        raise RuntimeError(
            "Unexpected channel feature count: "
            f"{len(channel_features)} instead of "
            f"{expected_channel_features}"
        )

    return channel_features


def extract_wrist_features(
    window_data: np.ndarray,
    wrist: str,
) -> dict[str, float]:

    expected_shape = (
        EXPECTED_WINDOW_SIZE,
        len(CHANNELS),
    )

    if window_data.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {wrist}: "
            f"{window_data.shape}; "
            f"expected {expected_shape}."
        )

    wrist_features: dict[str, float] = {}

    for channel_index, channel_name in enumerate(
        CHANNELS
    ):
        signal = window_data[
            :,
            channel_index,
        ]

        channel_features = (
            extract_channel_features(
                signal=signal,
                wrist=wrist,
                channel=channel_name,
            )
        )

        wrist_features.update(
            channel_features
        )

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
            "Unexpected total WPT feature count: "
            f"{len(features)} instead of "
            f"{len(FEATURE_COLUMNS)}"
        )

    feature_values = np.fromiter(
        features.values(),
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(feature_values)
    ):
        raise ValueError(
            "Extracted WPT features contain "
            "NaN or Inf."
        )

    return features


# ============================================================
# Feature manifest
# ============================================================

def create_feature_manifest() -> pd.DataFrame:

    descriptions = {
        "mean": (
            "Mean of WPT coefficients"
        ),
        "variance": (
            "Variance of WPT coefficients"
        ),
        "std": (
            "Standard deviation of WPT coefficients"
        ),
        "energy": (
            "Sum of squared WPT coefficients"
        ),
        "relative_energy": (
            "Band energy divided by total "
            "terminal-band channel energy"
        ),
        "entropy": (
            "Shannon entropy of normalized "
            "squared WPT coefficients"
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

    band_width_hz = (
        (FS / 2)
        / EXPECTED_TERMINAL_BANDS
    )

    rows: list[dict[str, Any]] = []

    for wrist in WRISTS:
        for channel in CHANNELS:
            for band_index, (
                band_name,
                node_path,
            ) in enumerate(
                zip(
                    BAND_NAMES,
                    TERMINAL_NODE_PATHS,
                )
            ):

                nominal_low_hz = (
                    band_index
                    * band_width_hz
                )

                nominal_high_hz = (
                    (band_index + 1)
                    * band_width_hz
                )

                for statistic in STATISTICS:
                    rows.append(
                        {
                            "feature_name": (
                                create_feature_name(
                                    wrist=wrist,
                                    channel=channel,
                                    band=band_name,
                                    statistic=statistic,
                                )
                            ),
                            "wrist": wrist,
                            "channel": channel,
                            "band_index": band_index,
                            "band_name": band_name,
                            "node_path": node_path,
                            "statistic": statistic,
                            "description": (
                                descriptions[
                                    statistic
                                ]
                            ),
                            "wavelet": WAVELET_NAME,
                            "wpt_level": WPT_LEVEL,
                            "boundary_mode": WPT_MODE,
                            "frequency_order": True,
                            "nominal_low_hz": (
                                nominal_low_hz
                            ),
                            "nominal_high_hz": (
                                nominal_high_hz
                            ),
                        }
                    )

    manifest = pd.DataFrame(rows)

    if len(manifest) != len(FEATURE_COLUMNS):
        raise RuntimeError(
            "Feature manifest size mismatch."
        )

    return manifest


# ============================================================
# Batch DataFrame preparation
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

    batch[text_columns] = (
        batch[text_columns].astype(str)
    )

    batch[integer_columns] = (
        batch[integer_columns].astype(
            np.int32
        )
    )

    batch[FEATURE_COLUMNS] = (
        batch[FEATURE_COLUMNS].astype(
            np.float32
        )
    )

    feature_array = batch[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(feature_array)
    ):
        invalid_count = int(
            feature_array.size
            - np.isfinite(
                feature_array
            ).sum()
        )

        raise ValueError(
            "WPT feature batch contains "
            f"{invalid_count} invalid values."
        )

    return batch


# ============================================================
# Load and cache one subject's signal tables
# ============================================================

def create_subject_signal_cache(
    mat_data: dict[str, Any],
    activities: list[str],
) -> dict[tuple[str, str], np.ndarray]:

    cache: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    for activity in activities:
        for wrist in WRISTS:

            table_name = (
                f"{activity}_{wrist}"
            )

            if table_name not in mat_data:
                raise KeyError(
                    f"Missing MAT variable: "
                    f"{table_name}"
                )

            table = mat_data[
                table_name
            ]

            if not isinstance(
                table,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"{table_name} is not "
                    "a pandas DataFrame."
                )

            validate_signal_table(
                table=table,
                table_name=table_name,
            )

            cache[
                (activity, wrist)
            ] = table_to_array(table)

    return cache


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not WINDOW_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            "Window inventory not found: "
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
        "subject_id",
        "file_name",
        "label",
        "activity",
        "window_id",
        "start_index",
        "end_index",
        "window_size",
    }

    missing_columns = (
        required_columns
        - set(window_inventory.columns)
    )

    if missing_columns:
        raise ValueError(
            "Window inventory is missing: "
            f"{sorted(missing_columns)}"
        )

    if len(window_inventory) != 15975:
        print(
            "Warning: expected 15975 windows, "
            f"found {len(window_inventory)}."
        )

    if FEATURE_OUTPUT_PATH.exists():
        FEATURE_OUTPUT_PATH.unlink()

    feature_manifest = (
        create_feature_manifest()
    )

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    grouped_files = (
        window_inventory.groupby(
            "file_name",
            sort=True,
        )
    )

    parquet_writer: (
        pq.ParquetWriter | None
    ) = None

    parquet_schema: (
        pa.Schema | None
    ) = None

    processed_subjects = 0
    processed_rows = 0

    class_counter: Counter[str] = Counter()
    activity_counter: Counter[str] = Counter()

    extraction_errors: list[
        dict[str, str]
    ] = []

    try:
        for file_name, subject_windows in tqdm(
            grouped_files,
            total=window_inventory[
                "file_name"
            ].nunique(),
            desc="Extracting WPT features",
            unit="subject",
        ):

            file_name = str(file_name)
            file_path = DATA_DIR / file_name

            subject_id = str(
                subject_windows[
                    "subject_id"
                ].iloc[0]
            )

            try:
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"MAT file not found: "
                        f"{file_path}"
                    )

                mat_data = load_from_mat(
                    file_path,
                    raw_data=False,
                    add_table_attrs=True,
                )

                activities = sorted(
                    subject_windows[
                        "activity"
                    ].astype(str).unique()
                )

                if len(activities) != 11:
                    raise ValueError(
                        "Unexpected activity count "
                        f"for {file_name}: "
                        f"{len(activities)}"
                    )

                signal_cache = (
                    create_subject_signal_cache(
                        mat_data=mat_data,
                        activities=activities,
                    )
                )

                batch_rows: list[
                    dict[str, Any]
                ] = []

                for window_row in (
                    subject_windows.itertuples(
                        index=False
                    )
                ):

                    activity = str(
                        window_row.activity
                    )

                    label = str(
                        window_row.label
                    )

                    start_index = int(
                        window_row.start_index
                    )

                    end_index = int(
                        window_row.end_index
                    )

                    window_size = int(
                        window_row.window_size
                    )

                    if (
                        window_size
                        != EXPECTED_WINDOW_SIZE
                    ):
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

                    left_array = signal_cache[
                        (
                            activity,
                            "LeftWrist",
                        )
                    ]

                    right_array = signal_cache[
                        (
                            activity,
                            "RightWrist",
                        )
                    ]

                    if end_index > len(left_array):
                        raise IndexError(
                            "Left-wrist window exceeds "
                            "signal length."
                        )

                    if end_index > len(right_array):
                        raise IndexError(
                            "Right-wrist window exceeds "
                            "signal length."
                        )

                    left_window = np.array(
                        left_array[
                            start_index:end_index,
                            :,
                        ],
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )

                    right_window = np.array(
                        right_array[
                            start_index:end_index,
                            :,
                        ],
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )

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

                    batch_rows.append(
                        result_row
                    )

                batch_dataframe = (
                    prepare_batch_dataframe(
                        batch_rows
                    )
                )

                arrow_table = pa.Table.from_pandas(
                    batch_dataframe,
                    preserve_index=False,
                )

                if parquet_writer is None:
                    parquet_schema = (
                        arrow_table.schema
                    )

                    parquet_writer = (
                        pq.ParquetWriter(
                            where=str(
                                FEATURE_OUTPUT_PATH
                            ),
                            schema=parquet_schema,
                            compression="snappy",
                        )
                    )

                elif (
                    arrow_table.schema
                    != parquet_schema
                ):
                    arrow_table = (
                        arrow_table.cast(
                            parquet_schema
                        )
                    )

                parquet_writer.write_table(
                    arrow_table
                )

                processed_subjects += 1
                processed_rows += len(
                    batch_dataframe
                )

                class_counter.update(
                    batch_dataframe[
                        "label"
                    ].tolist()
                )

                activity_counter.update(
                    batch_dataframe[
                        "activity"
                    ].tolist()
                )

            except Exception as error:
                extraction_errors.append(
                    {
                        "subject_id": subject_id,
                        "file_name": file_name,
                        "error_type": (
                            type(error).__name__
                        ),
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

    print("\n" + "=" * 72)
    print("WPT FEATURE EXTRACTION SUMMARY")
    print("=" * 72)

    print(f"\nWavelet: {WAVELET_NAME}")
    print(f"WPT level: {WPT_LEVEL}")
    print(f"Boundary mode: {WPT_MODE}")
    print(
        "Terminal bands: "
        f"{EXPECTED_TERMINAL_BANDS}"
    )

    print(
        "\nFrequency-ordered node paths:"
    )

    for band_index, (
        band_name,
        node_path,
    ) in enumerate(
        zip(
            BAND_NAMES,
            TERMINAL_NODE_PATHS,
        )
    ):
        print(
            f"{band_index:02d}: "
            f"{band_name} "
            f"(path={node_path})"
        )

    print(
        f"\nProcessed subjects: "
        f"{processed_subjects}"
    )

    print(
        f"Created feature rows: "
        f"{processed_rows}"
    )

    print(
        "WPT features per row: "
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

    for label, count in sorted(
        class_counter.items()
    ):
        print(f"{label}: {count}")

    print("\nRows per activity:")

    for activity in sorted(
        activity_counter
    ):
        print(
            f"{activity}: "
            f"{activity_counter[activity]}"
        )

    print("\nCreated files:")
    print(FEATURE_OUTPUT_PATH)
    print(FEATURE_MANIFEST_PATH)
    print(ERROR_OUTPUT_PATH)

    if processed_rows != len(
        window_inventory
    ):
        raise RuntimeError(
            "Extracted row count does not match "
            "window inventory. "
            f"Expected {len(window_inventory)}, "
            f"created {processed_rows}."
        )

    if processed_subjects != 355:
        raise RuntimeError(
            "Processed subject count mismatch. "
            f"Expected 355, "
            f"processed {processed_subjects}."
        )

    if not error_dataframe.empty:
        raise RuntimeError(
            "WPT extraction completed with errors. "
            f"Inspect: {ERROR_OUTPUT_PATH}"
        )


if __name__ == "__main__":
    main()