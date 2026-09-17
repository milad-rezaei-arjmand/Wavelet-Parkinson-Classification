from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

WPT_FEATURE_PATH = (
    OUTPUTS_DIR / "wpt_features_db4_level4.parquet"
)

RESULTS_DIR = (
    OUTPUTS_DIR / "subject_level_wpt_screening"
)

FIGURES_DIR = RESULTS_DIR / "figures"

SUBJECT_FEATURE_MANIFEST_PATH = (
    RESULTS_DIR / "subject_feature_manifest.csv"
)

FOLD_RESULTS_PATH = (
    RESULTS_DIR / "method_fold_results.csv"
)

OVERALL_RESULTS_PATH = (
    RESULTS_DIR / "method_overall_results.csv"
)

SUBJECT_PREDICTIONS_PATH = (
    RESULTS_DIR / "subject_predictions.csv"
)

SELECTED_FEATURES_PATH = (
    RESULTS_DIR / "selected_feature_frequency.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

SELECT_K = 100

EPSILON = 1e-12

LABEL_TO_ID = {
    "Healthy": 0,
    "Parkinson": 1,
}

ID_TO_LABEL = {
    0: "Healthy",
    1: "Parkinson",
}

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


# ============================================================
# Data loading
# ============================================================

def normalize_subject_ids(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(3)
    )

    return dataframe


def load_window_level_wpt() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:

    if not WPT_FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"WPT feature file not found: "
            f"{WPT_FEATURE_PATH}"
        )

    print("Loading WPT features...")

    data = pd.read_parquet(
        WPT_FEATURE_PATH
    )

    missing_metadata = (
        set(META_COLUMNS)
        - set(data.columns)
    )

    if missing_metadata:
        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    data = normalize_subject_ids(
        data
    )

    data = (
        data.sort_values(
            [
                "subject_id",
                "activity",
                "window_id",
                "start_index",
            ]
        )
        .reset_index(drop=True)
    )

    if len(data) != 15975:
        raise ValueError(
            "Unexpected row count: "
            f"{len(data)}"
        )

    if data["subject_id"].nunique() != 355:
        raise ValueError(
            "Expected exactly 355 subjects."
        )

    if data["activity"].nunique() != 11:
        raise ValueError(
            "Expected exactly 11 activities."
        )

    unknown_labels = (
        set(data["label"].unique())
        - set(LABEL_TO_ID)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {unknown_labels}"
        )

    windows_per_subject = (
        data.groupby("subject_id").size()
    )

    if not np.all(
        windows_per_subject == 45
    ):
        raise ValueError(
            "Not every subject has 45 windows."
        )

    activities_per_subject = (
        data.groupby("subject_id")[
            "activity"
        ].nunique()
    )

    if not np.all(
        activities_per_subject == 11
    ):
        raise ValueError(
            "Not every subject has 11 activities."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in META_COLUMNS
    ]

    if len(feature_columns) != 1728:
        raise ValueError(
            "Unexpected WPT feature count: "
            f"{len(feature_columns)}"
        )

    print("Converting WPT features to NumPy...")

    X_window = data[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.all(
        np.isfinite(X_window)
    ):
        raise ValueError(
            "WPT matrix contains NaN or Inf."
        )

    y_window = (
        data["label"]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    groups_window = (
        data["subject_id"]
        .to_numpy()
    )

    metadata = data[
        META_COLUMNS
    ].copy()

    del data
    gc.collect()

    return (
        metadata,
        X_window,
        y_window,
        groups_window,
        feature_columns,
    )


# ============================================================
# Subject-level feature construction
# ============================================================

def create_subject_level_representations(
    metadata: pd.DataFrame,
    X_window: np.ndarray,
    y_window: np.ndarray,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, list[str]],
]:

    subjects = sorted(
        metadata["subject_id"]
        .astype(str)
        .unique()
    )

    activities = sorted(
        metadata["activity"]
        .astype(str)
        .unique()
    )

    if len(subjects) != 355:
        raise ValueError(
            "Expected 355 subjects."
        )

    if len(activities) != 11:
        raise ValueError(
            "Expected 11 activities."
        )

    subject_rows: list[
        dict[str, Any]
    ] = []

    global_mean_std_rows: list[
        np.ndarray
    ] = []

    activity_mean_rows: list[
        np.ndarray
    ] = []

    for subject_id in subjects:

        subject_mask = (
            metadata[
                "subject_id"
            ].astype(str).to_numpy()
            == subject_id
        )

        subject_indices = np.flatnonzero(
            subject_mask
        )

        if len(subject_indices) != 45:
            raise ValueError(
                f"Subject {subject_id} has "
                f"{len(subject_indices)} windows."
            )

        subject_labels = np.unique(
            y_window[
                subject_indices
            ]
        )

        if len(subject_labels) != 1:
            raise ValueError(
                f"Inconsistent labels for "
                f"subject {subject_id}."
            )

        subject_X = X_window[
            subject_indices
        ]

        global_mean = np.mean(
            subject_X,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        global_std = np.std(
            subject_X,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        global_mean_std = np.concatenate(
            [
                global_mean,
                global_std,
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        activity_vectors: list[
            np.ndarray
        ] = []

        subject_activities = (
            metadata.iloc[
                subject_indices
            ]["activity"]
            .astype(str)
            .to_numpy()
        )

        for activity in activities:

            relative_indices = np.flatnonzero(
                subject_activities
                == activity
            )

            if len(relative_indices) not in {
                3,
                7,
            }:
                raise ValueError(
                    f"Subject {subject_id}, "
                    f"activity {activity}: "
                    f"unexpected window count "
                    f"{len(relative_indices)}."
                )

            activity_mean = np.mean(
                subject_X[
                    relative_indices
                ],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

            activity_vectors.append(
                activity_mean
            )

        activity_mean_vector = (
            np.concatenate(
                activity_vectors
            ).astype(
                np.float32,
                copy=False,
            )
        )

        global_mean_std_rows.append(
            global_mean_std
        )

        activity_mean_rows.append(
            activity_mean_vector
        )

        subject_rows.append(
            {
                "subject_id": subject_id,
                "y_true": int(
                    subject_labels[0]
                ),
                "label": ID_TO_LABEL[
                    int(subject_labels[0])
                ],
            }
        )

    X_global = np.vstack(
        global_mean_std_rows
    ).astype(
        np.float32,
        copy=False,
    )

    X_activity = np.vstack(
        activity_mean_rows
    ).astype(
        np.float32,
        copy=False,
    )

    X_fusion = np.concatenate(
        [
            X_global,
            X_activity,
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if X_global.shape != (
        355,
        3456,
    ):
        raise RuntimeError(
            "Unexpected global representation "
            f"shape: {X_global.shape}"
        )

    if X_activity.shape != (
        355,
        19008,
    ):
        raise RuntimeError(
            "Unexpected activity representation "
            f"shape: {X_activity.shape}"
        )

    if X_fusion.shape != (
        355,
        22464,
    ):
        raise RuntimeError(
            "Unexpected fusion representation "
            f"shape: {X_fusion.shape}"
        )

    for representation_name, matrix in {
        "Subject_GlobalMeanStd": X_global,
        "Subject_ActivityMean": X_activity,
        "Subject_GlobalActivityFusion": X_fusion,
    }.items():

        if not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                f"{representation_name} contains "
                "NaN or Inf."
            )

    global_feature_names = (
        [
            f"GLOBAL_MEAN__{feature}"
            for feature in feature_columns
        ]
        + [
            f"GLOBAL_STD__{feature}"
            for feature in feature_columns
        ]
    )

    activity_feature_names: list[
        str
    ] = []

    for activity in activities:
        activity_feature_names.extend(
            [
                f"ACTIVITY_MEAN__{activity}__{feature}"
                for feature in feature_columns
            ]
        )

    representation_feature_names = {
        "Subject_GlobalMeanStd": (
            global_feature_names
        ),
        "Subject_ActivityMean": (
            activity_feature_names
        ),
        "Subject_GlobalActivityFusion": (
            global_feature_names
            + activity_feature_names
        ),
    }

    subject_table = pd.DataFrame(
        subject_rows
    )

    representations = {
        "Subject_GlobalMeanStd": X_global,
        "Subject_ActivityMean": X_activity,
        "Subject_GlobalActivityFusion": X_fusion,
    }

    return (
        subject_table,
        representations,
        representation_feature_names,
    )


def create_subject_feature_manifest(
    representation_feature_names: dict[
        str,
        list[str],
    ],
) -> pd.DataFrame:

    rows: list[
        dict[str, Any]
    ] = []

    for (
        representation,
        feature_names,
    ) in representation_feature_names.items():

        for feature_index, feature_name in enumerate(
            feature_names
        ):
            if feature_name.startswith(
                "GLOBAL_MEAN__"
            ):
                source = "global_mean"
                activity = ""

            elif feature_name.startswith(
                "GLOBAL_STD__"
            ):
                source = "global_std"
                activity = ""

            elif feature_name.startswith(
                "ACTIVITY_MEAN__"
            ):
                source = "activity_mean"

                parts = feature_name.split(
                    "__",
                    maxsplit=2,
                )

                activity = parts[1]

            else:
                source = "unknown"
                activity = ""

            rows.append(
                {
                    "representation": (
                        representation
                    ),
                    "feature_index": (
                        feature_index
                    ),
                    "feature_name": (
                        feature_name
                    ),
                    "source": source,
                    "activity": activity,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Fold construction matching window-level protocol
# ============================================================

def create_outer_subject_splits(
    y_window: np.ndarray,
    groups_window: np.ndarray,
    subject_table: pd.DataFrame,
) -> list[
    tuple[np.ndarray, np.ndarray]
]:

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    dummy_X = np.zeros(
        (
            len(y_window),
            1,
        ),
        dtype=np.float32,
    )

    subject_to_row = {
        subject_id: row_index
        for row_index, subject_id
        in enumerate(
            subject_table[
                "subject_id"
            ].astype(str)
        )
    }

    subject_splits: list[
        tuple[np.ndarray, np.ndarray]
    ] = []

    for (
        train_window_indices,
        test_window_indices,
    ) in outer_cv.split(
        X=dummy_X,
        y=y_window,
        groups=groups_window,
    ):

        train_subjects = sorted(
            set(
                groups_window[
                    train_window_indices
                ]
            )
        )

        test_subjects = sorted(
            set(
                groups_window[
                    test_window_indices
                ]
            )
        )

        if set(train_subjects).intersection(
            test_subjects
        ):
            raise RuntimeError(
                "Subject leakage in outer CV."
            )

        train_subject_indices = np.asarray(
            [
                subject_to_row[
                    str(subject_id)
                ]
                for subject_id
                in train_subjects
            ],
            dtype=np.int32,
        )

        test_subject_indices = np.asarray(
            [
                subject_to_row[
                    str(subject_id)
                ]
                for subject_id
                in test_subjects
            ],
            dtype=np.int32,
        )

        subject_splits.append(
            (
                train_subject_indices,
                test_subject_indices,
            )
        )

    return subject_splits


def create_inner_subject_splits(
    outer_train_subject_indices: np.ndarray,
    subject_table: pd.DataFrame,
    outer_fold: int,
) -> list[
    tuple[np.ndarray, np.ndarray]
]:

    outer_train_subjects = (
        subject_table.iloc[
            outer_train_subject_indices
        ].reset_index()
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=(
            RANDOM_STATE + outer_fold
        ),
    )

    y_inner = (
        outer_train_subjects[
            "y_true"
        ].to_numpy(
            dtype=np.int8
        )
    )

    groups_inner = (
        outer_train_subjects[
            "subject_id"
        ].astype(str).to_numpy()
    )

    dummy_X = np.zeros(
        (
            len(
                outer_train_subjects
            ),
            1,
        ),
        dtype=np.float32,
    )

    inner_splits: list[
        tuple[np.ndarray, np.ndarray]
    ] = []

    for (
        inner_train_relative,
        inner_validation_relative,
    ) in inner_cv.split(
        X=dummy_X,
        y=y_inner,
        groups=groups_inner,
    ):

        inner_train_indices = (
            outer_train_subject_indices[
                inner_train_relative
            ]
        )

        inner_validation_indices = (
            outer_train_subject_indices[
                inner_validation_relative
            ]
        )

        train_subjects = set(
            subject_table.iloc[
                inner_train_indices
            ]["subject_id"]
        )

        validation_subjects = set(
            subject_table.iloc[
                inner_validation_indices
            ]["subject_id"]
        )

        if train_subjects.intersection(
            validation_subjects
        ):
            raise RuntimeError(
                "Subject leakage in inner CV."
            )

        inner_splits.append(
            (
                inner_train_indices,
                inner_validation_indices,
            )
        )

    return inner_splits


# ============================================================
# Models
# ============================================================

def create_models() -> dict[
    str,
    Pipeline,
]:

    common_steps = [
        (
            "variance",
            VarianceThreshold(
                threshold=0.0
            ),
        ),
        (
            "select",
            SelectKBest(
                score_func=f_classif,
                k=SELECT_K,
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
    ]

    return {
        "LinearSVM": Pipeline(
            steps=[
                *common_steps,
                (
                    "classifier",
                    LinearSVC(
                        C=0.1,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=20000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RBF_SVM": Pipeline(
            steps=[
                *common_steps,
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        cache_size=1024,
                    ),
                ),
            ]
        ),
    }


def get_decision_scores(
    fitted_model: Pipeline,
    X: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(
        fitted_model.decision_function(X),
        dtype=float,
    ).reshape(-1)

    if not np.all(
        np.isfinite(scores)
    ):
        raise ValueError(
            "Decision scores contain "
            "NaN or Inf."
        )

    return scores


# ============================================================
# Metrics and threshold selection
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "healthy_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                average="binary",
                zero_division=0,
            )
        ),
        "healthy_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                average="binary",
                zero_division=0,
            )
        ),
        "healthy_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                average="binary",
                zero_division=0,
            )
        ),
        "parkinson_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                average="binary",
                zero_division=0,
            )
        ),
        "parkinson_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                average="binary",
                zero_division=0,
            )
        ),
        "parkinson_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                average="binary",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
    }


def create_threshold_candidates(
    scores: np.ndarray,
) -> np.ndarray:

    unique_scores = np.unique(
        np.asarray(
            scores,
            dtype=float,
        )
    )

    if unique_scores.size == 0:
        raise ValueError(
            "No scores available."
        )

    if unique_scores.size == 1:
        return np.array(
            [
                unique_scores[0] - 1e-6,
                unique_scores[0],
                unique_scores[0] + 1e-6,
            ],
            dtype=float,
        )

    middle_points = (
        unique_scores[:-1]
        + unique_scores[1:]
    ) / 2.0

    return np.unique(
        np.concatenate(
            [
                [
                    unique_scores[0]
                    - 1e-6
                ],
                middle_points,
                [
                    unique_scores[-1]
                    + 1e-6
                ],
                [0.0],
            ]
        )
    )


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[
    float,
    dict[str, float],
]:

    best_threshold: float | None = None
    best_metrics: dict[
        str,
        float
    ] | None = None
    best_key: tuple[
        float,
        ...
    ] | None = None

    for threshold in (
        create_threshold_candidates(
            scores
        )
    ):

        y_pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
        )

        minimum_class_recall = min(
            metrics["healthy_recall"],
            metrics["parkinson_recall"],
        )

        selection_key = (
            metrics["macro_f1"],
            minimum_class_recall,
            metrics["balanced_accuracy"],
            metrics["healthy_recall"],
            metrics["roc_auc"],
            -abs(float(threshold)),
        )

        if (
            best_key is None
            or selection_key > best_key
        ):
            best_key = selection_key
            best_threshold = float(
                threshold
            )
            best_metrics = metrics

    if (
        best_threshold is None
        or best_metrics is None
    ):
        raise RuntimeError(
            "Threshold selection failed."
        )

    return (
        best_threshold,
        best_metrics,
    )


# ============================================================
# Feature tracking
# ============================================================

def get_selected_feature_names(
    fitted_model: Pipeline,
    original_feature_names: list[str],
) -> list[str]:

    variance_selector = (
        fitted_model.named_steps[
            "variance"
        ]
    )

    feature_selector = (
        fitted_model.named_steps[
            "select"
        ]
    )

    after_variance = np.asarray(
        original_feature_names
    )[
        variance_selector.get_support()
    ]

    selected_names = (
        after_variance[
            feature_selector.get_support()
        ]
    )

    return selected_names.tolist()


def infer_feature_source(
    feature_name: str,
) -> tuple[str, str]:

    if feature_name.startswith(
        "GLOBAL_MEAN__"
    ):
        return (
            "global_mean",
            "",
        )

    if feature_name.startswith(
        "GLOBAL_STD__"
    ):
        return (
            "global_std",
            "",
        )

    if feature_name.startswith(
        "ACTIVITY_MEAN__"
    ):
        parts = feature_name.split(
            "__",
            maxsplit=2,
        )

        return (
            "activity_mean",
            parts[1],
        )

    return (
        "unknown",
        "",
    )


# ============================================================
# Output utilities
# ============================================================

def safe_file_name(
    text: str,
) -> str:

    return (
        text.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(5.5, 4.5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Healthy",
            "Parkinson",
        ],
    )

    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        method_name
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / (
            safe_file_name(
                method_name
            )
            + "_confusion_matrix.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str,
) -> None:

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Healthy",
            "Parkinson",
        ],
        digits=4,
        zero_division=0,
    )

    output_path = (
        RESULTS_DIR
        / (
            safe_file_name(
                method_name
            )
            + "_classification_report.txt"
        )
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        metadata_window,
        X_window,
        y_window,
        groups_window,
        feature_columns,
    ) = load_window_level_wpt()

    print(
        "Creating subject-level "
        "representations..."
    )

    (
        subject_table,
        representations,
        representation_feature_names,
    ) = create_subject_level_representations(
        metadata=metadata_window,
        X_window=X_window,
        y_window=y_window,
        feature_columns=feature_columns,
    )

    feature_manifest = (
        create_subject_feature_manifest(
            representation_feature_names
        )
    )

    feature_manifest.to_csv(
        SUBJECT_FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    y_subject = (
        subject_table[
            "y_true"
        ].to_numpy(
            dtype=np.int8
        )
    )

    outer_splits = (
        create_outer_subject_splits(
            y_window=y_window,
            groups_window=groups_window,
            subject_table=subject_table,
        )
    )

    models = create_models()

    print("\n" + "=" * 88)
    print("SUBJECT-LEVEL WPT AGGREGATION SCREENING")
    print("=" * 88)

    print(f"\nSubjects: {len(subject_table)}")
    print(f"Selected features: {SELECT_K}")
    print(f"Outer folds: {N_OUTER_SPLITS}")
    print(f"Inner folds: {N_INNER_SPLITS}")

    for (
        representation_name,
        X_subject,
    ) in representations.items():

        print(
            f"{representation_name}: "
            f"{X_subject.shape[1]} features"
        )

    print(
        "Models: "
        + ", ".join(
            models.keys()
        )
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    feature_counters: dict[
        str,
        Counter
    ] = {}

    for representation_name in representations:
        for model_name in models:
            method_name = (
                representation_name
                + "__"
                + model_name
            )

            feature_counters[
                method_name
            ] = Counter()

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        print("\n" + "=" * 88)
        print(
            f"OUTER FOLD "
            f"{outer_fold}/{N_OUTER_SPLITS}"
        )
        print("=" * 88)

        print(
            "Train subjects: "
            f"{len(outer_train_indices)}"
        )

        print(
            "Test subjects: "
            f"{len(outer_test_indices)}"
        )

        inner_splits = (
            create_inner_subject_splits(
                outer_train_subject_indices=(
                    outer_train_indices
                ),
                subject_table=subject_table,
                outer_fold=outer_fold,
            )
        )

        for (
            representation_name,
            X_subject,
        ) in representations.items():

            for (
                model_name,
                model_template,
            ) in models.items():

                method_name = (
                    representation_name
                    + "__"
                    + model_name
                )

                print(
                    "\nMethod: "
                    f"{method_name}"
                )

                inner_scores = np.full(
                    len(subject_table),
                    np.nan,
                    dtype=float,
                )

                for (
                    inner_train_indices,
                    inner_validation_indices,
                ) in inner_splits:

                    model = clone(
                        model_template
                    )

                    model.fit(
                        X_subject[
                            inner_train_indices
                        ],
                        y_subject[
                            inner_train_indices
                        ],
                    )

                    validation_scores = (
                        get_decision_scores(
                            fitted_model=model,
                            X=X_subject[
                                inner_validation_indices
                            ],
                        )
                    )

                    inner_scores[
                        inner_validation_indices
                    ] = validation_scores

                    del model
                    gc.collect()

                outer_train_scores = (
                    inner_scores[
                        outer_train_indices
                    ]
                )

                if not np.all(
                    np.isfinite(
                        outer_train_scores
                    )
                ):
                    raise RuntimeError(
                        "Incomplete inner OOF scores."
                    )

                inner_y_true = (
                    y_subject[
                        outer_train_indices
                    ]
                )

                (
                    selected_threshold,
                    inner_metrics,
                ) = find_best_threshold(
                    y_true=inner_y_true,
                    scores=outer_train_scores,
                )

                inner_score_std = float(
                    np.std(
                        outer_train_scores
                    )
                )

                if (
                    inner_score_std
                    <= EPSILON
                ):
                    inner_score_std = 1.0

                final_model = clone(
                    model_template
                )

                final_model.fit(
                    X_subject[
                        outer_train_indices
                    ],
                    y_subject[
                        outer_train_indices
                    ],
                )

                outer_scores = (
                    get_decision_scores(
                        fitted_model=final_model,
                        X=X_subject[
                            outer_test_indices
                        ],
                    )
                )

                outer_y_true = (
                    y_subject[
                        outer_test_indices
                    ]
                )

                outer_y_pred = (
                    outer_scores
                    >= selected_threshold
                ).astype(np.int8)

                normalized_margin = (
                    (
                        outer_scores
                        - selected_threshold
                    )
                    / inner_score_std
                )

                outer_metrics = (
                    calculate_metrics(
                        y_true=outer_y_true,
                        y_pred=outer_y_pred,
                        scores=outer_scores,
                    )
                )

                selected_features = (
                    get_selected_feature_names(
                        fitted_model=final_model,
                        original_feature_names=(
                            representation_feature_names[
                                representation_name
                            ]
                        ),
                    )
                )

                feature_counters[
                    method_name
                ].update(
                    selected_features
                )

                fold_rows.append(
                    {
                        "method": method_name,
                        "representation": (
                            representation_name
                        ),
                        "model": model_name,
                        "outer_fold": (
                            outer_fold
                        ),
                        "train_subjects": len(
                            outer_train_indices
                        ),
                        "test_subjects": len(
                            outer_test_indices
                        ),
                        "raw_feature_count": (
                            X_subject.shape[1]
                        ),
                        "selected_feature_count": (
                            len(
                                selected_features
                            )
                        ),
                        "selected_threshold": (
                            selected_threshold
                        ),
                        "inner_score_std": (
                            inner_score_std
                        ),
                        **{
                            f"inner_{key}": value
                            for key, value
                            in inner_metrics.items()
                        },
                        **{
                            f"outer_{key}": value
                            for key, value
                            in outer_metrics.items()
                        },
                    }
                )

                prediction_frame = (
                    subject_table.iloc[
                        outer_test_indices
                    ][
                        [
                            "subject_id",
                            "y_true",
                            "label",
                        ]
                    ].copy()
                )

                prediction_frame[
                    "method"
                ] = method_name

                prediction_frame[
                    "representation"
                ] = representation_name

                prediction_frame[
                    "model"
                ] = model_name

                prediction_frame[
                    "outer_fold"
                ] = outer_fold

                prediction_frame[
                    "score"
                ] = outer_scores

                prediction_frame[
                    "normalized_margin"
                ] = normalized_margin

                prediction_frame[
                    "selected_threshold"
                ] = selected_threshold

                prediction_frame[
                    "y_pred"
                ] = outer_y_pred

                prediction_frame[
                    "predicted_label"
                ] = [
                    ID_TO_LABEL[
                        int(value)
                    ]
                    for value in outer_y_pred
                ]

                prediction_frames.append(
                    prediction_frame
                )

                print(
                    "Selected threshold: "
                    f"{selected_threshold:.6f}"
                )

                print(
                    "Inner Macro-F1:     "
                    f"{inner_metrics['macro_f1']:.4f}"
                )

                print(
                    "Outer Accuracy:     "
                    f"{outer_metrics['accuracy']:.4f}"
                )

                print(
                    "Outer Balanced Acc: "
                    f"{outer_metrics['balanced_accuracy']:.4f}"
                )

                print(
                    "Outer Macro-F1:     "
                    f"{outer_metrics['macro_f1']:.4f}"
                )

                print(
                    "Healthy Recall:     "
                    f"{outer_metrics['healthy_recall']:.4f}"
                )

                print(
                    "Parkinson Recall:   "
                    f"{outer_metrics['parkinson_recall']:.4f}"
                )

                print(
                    "ROC-AUC:            "
                    f"{outer_metrics['roc_auc']:.4f}"
                )

                del final_model
                gc.collect()

    fold_table = pd.DataFrame(
        fold_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    expected_prediction_rows = (
        355
        * len(representations)
        * len(models)
    )

    if (
        len(predictions)
        != expected_prediction_rows
    ):
        raise RuntimeError(
            "Unexpected prediction count: "
            f"{len(predictions)}"
        )

    overall_rows: list[
        dict[str, Any]
    ] = []

    for method_name in sorted(
        predictions[
            "method"
        ].unique()
    ):

        current = predictions[
            predictions[
                "method"
            ] == method_name
        ].copy()

        if len(current) != 355:
            raise RuntimeError(
                "Each method must predict "
                "355 subjects."
            )

        if (
            current[
                "subject_id"
            ].nunique()
            != 355
        ):
            raise RuntimeError(
                "Subjects duplicated or missing."
            )

        metrics = calculate_metrics(
            y_true=current[
                "y_true"
            ].to_numpy(
                dtype=np.int8
            ),
            y_pred=current[
                "y_pred"
            ].to_numpy(
                dtype=np.int8
            ),
            scores=current[
                "normalized_margin"
            ].to_numpy(
                dtype=float
            ),
        )

        current_folds = fold_table[
            fold_table[
                "method"
            ] == method_name
        ]

        first_row = (
            current_folds.iloc[0]
        )

        overall_rows.append(
            {
                "method": method_name,
                "representation": (
                    first_row[
                        "representation"
                    ]
                ),
                "model": first_row[
                    "model"
                ],
                "n_subjects": 355,
                "raw_feature_count": int(
                    first_row[
                        "raw_feature_count"
                    ]
                ),
                "selected_feature_count": (
                    SELECT_K
                ),
                **metrics,
                "fold_accuracy_mean": float(
                    current_folds[
                        "outer_accuracy"
                    ].mean()
                ),
                "fold_accuracy_std": float(
                    current_folds[
                        "outer_accuracy"
                    ].std()
                ),
                "fold_macro_f1_mean": float(
                    current_folds[
                        "outer_macro_f1"
                    ].mean()
                ),
                "fold_macro_f1_std": float(
                    current_folds[
                        "outer_macro_f1"
                    ].std()
                ),
                "fold_roc_auc_mean": float(
                    current_folds[
                        "outer_roc_auc"
                    ].mean()
                ),
                "fold_roc_auc_std": float(
                    current_folds[
                        "outer_roc_auc"
                    ].std()
                ),
            }
        )

        save_confusion_matrix(
            y_true=current[
                "y_true"
            ].to_numpy(
                dtype=np.int8
            ),
            y_pred=current[
                "y_pred"
            ].to_numpy(
                dtype=np.int8
            ),
            method_name=method_name,
        )

        save_classification_report(
            y_true=current[
                "y_true"
            ].to_numpy(
                dtype=np.int8
            ),
            y_pred=current[
                "y_pred"
            ].to_numpy(
                dtype=np.int8
            ),
            method_name=method_name,
        )

    overall_table = pd.DataFrame(
        overall_rows
    ).sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=False,
    )

    selected_feature_rows: list[
        dict[str, Any]
    ] = []

    for (
        method_name,
        counter,
    ) in feature_counters.items():

        for (
            feature_name,
            selected_folds,
        ) in counter.most_common():

            (
                feature_source,
                activity,
            ) = infer_feature_source(
                feature_name
            )

            selected_feature_rows.append(
                {
                    "method": method_name,
                    "feature_source": (
                        feature_source
                    ),
                    "activity": activity,
                    "feature_name": (
                        feature_name
                    ),
                    "selected_folds": (
                        selected_folds
                    ),
                    "selection_rate": (
                        selected_folds
                        / N_OUTER_SPLITS
                    ),
                }
            )

    selected_feature_table = pd.DataFrame(
        selected_feature_rows
    )

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overall_table.to_csv(
        OVERALL_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    selected_feature_table.to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 104)
    print("OVERALL SUBJECT-LEVEL WPT RESULTS")
    print("=" * 104)

    print(
        overall_table[
            [
                "method",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "healthy_recall",
                "parkinson_recall",
                "roc_auc",
                "fold_roc_auc_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    best_method = str(
        overall_table.iloc[0][
            "method"
        ]
    )

    print(
        "\nBest method selected-feature sources:"
    )

    best_sources = (
        selected_feature_table[
            selected_feature_table[
                "method"
            ] == best_method
        ]
        .groupby(
            [
                "feature_source",
                "activity",
            ],
            dropna=False,
        )["selected_folds"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print(
        best_sources.to_string()
    )

    print("\nCreated files:")

    for output_path in [
        SUBJECT_FEATURE_MANIFEST_PATH,
        FOLD_RESULTS_PATH,
        OVERALL_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        SELECTED_FEATURES_PATH,
        FIGURES_DIR,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
