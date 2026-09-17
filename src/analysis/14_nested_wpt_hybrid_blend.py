from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

WPT_SCRIPT = PROJECT_ROOT / "12_tune_wpt_svm.py"
ACTIVITY_SCRIPT = PROJECT_ROOT / "13_nested_wpt_activity_ensemble.py"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "nested_wpt_hybrid_blend"
)

FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
BLEND_SEARCH_PATH = OUTPUT_DIR / "blend_search.csv"
SUBJECT_PREDICTIONS_PATH = OUTPUT_DIR / "subject_predictions.csv"
ACTIVITY_SELECTION_PATH = OUTPUT_DIR / "activity_selection.csv"
OVERALL_SUMMARY_PATH = OUTPUT_DIR / "overall_summary.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "subject_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
EPSILON = 1e-12

# Honest fixed configuration selected from the WPT representation screening.
BASE_CONFIGURATION = {
    "selected_features": 300,
    "C": 10.0,
    "gamma": "scale",
    "configuration_id": 0,
}

# Weight assigned to the global WPT stream.
# Activity-stream weight is 1 - global_weight.
GLOBAL_WEIGHT_CANDIDATES = np.linspace(
    0.0,
    1.0,
    11,
).tolist()

ID_TO_LABEL = {
    0: "Healthy",
    1: "Parkinson",
}


# ============================================================
# Module loading
# ============================================================

def load_python_module(
    file_path: Path,
    module_name: str,
):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Python file not found: {file_path}"
        )

    specification = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"Could not load Python module: {file_path}"
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# ============================================================
# Validation and normalization
# ============================================================

def safe_standard_deviation(values: np.ndarray) -> float:
    value = float(np.std(np.asarray(values, dtype=float)))

    if not np.isfinite(value) or value <= EPSILON:
        return 1.0

    return value


def standardize_scores(
    scores: np.ndarray,
    mean_value: float,
    std_value: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    standardized = (
        scores - float(mean_value)
    ) / float(std_value)

    if not np.all(np.isfinite(standardized)):
        raise ValueError(
            "Standardized stream scores contain NaN or Inf."
        )

    return standardized


def assert_subject_table(
    table: pd.DataFrame,
    expected_subjects: set[str],
    table_name: str,
) -> None:
    if len(table) != len(expected_subjects):
        raise RuntimeError(
            f"{table_name} has {len(table)} rows; "
            f"expected {len(expected_subjects)}."
        )

    actual_subjects = set(
        table["subject_id"].astype(str)
    )

    if actual_subjects != expected_subjects:
        missing = sorted(expected_subjects - actual_subjects)
        extra = sorted(actual_subjects - expected_subjects)

        raise RuntimeError(
            f"Invalid subject set in {table_name}. "
            f"Missing={missing}, extra={extra}"
        )

    counts = table["subject_id"].value_counts()

    if not np.all(counts == 1):
        raise RuntimeError(
            f"Duplicated subjects in {table_name}."
        )


# ============================================================
# Global WPT stream
# ============================================================

def get_inner_global_oof_scores(
    wpt: Any,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    inner_splits_relative: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> pd.DataFrame:
    """Create one global-WPT OOF score per outer-train subject."""

    frames: list[pd.DataFrame] = []

    for inner_fold, (
        inner_train_relative,
        inner_validation_relative,
    ) in enumerate(
        inner_splits_relative,
        start=1,
    ):
        inner_train_indices = (
            outer_train_indices[
                inner_train_relative
            ]
        )

        inner_validation_indices = (
            outer_train_indices[
                inner_validation_relative
            ]
        )

        train_subjects = set(
            groups[inner_train_indices]
        )

        validation_subjects = set(
            groups[inner_validation_indices]
        )

        if train_subjects.intersection(
            validation_subjects
        ):
            raise RuntimeError(
                "Subject leakage in inner global WPT CV."
            )

        model = wpt.create_model(
            BASE_CONFIGURATION
        )

        model.fit(
            X[inner_train_indices],
            y[inner_train_indices],
        )

        validation_window_scores = (
            wpt.get_decision_scores(
                fitted_model=model,
                X=X[
                    inner_validation_indices
                ],
            )
        )

        validation_subjects_table = (
            wpt.aggregate_subject_scores(
                metadata=metadata.iloc[
                    inner_validation_indices
                ],
                y_true=y[
                    inner_validation_indices
                ],
                scores=validation_window_scores,
            )
        )

        validation_subjects_table[
            "inner_fold"
        ] = inner_fold

        frames.append(
            validation_subjects_table
        )

        del model
        gc.collect()

    global_oof = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_subjects = set(
        groups[outer_train_indices]
    )

    assert_subject_table(
        global_oof,
        expected_subjects,
        "inner global OOF scores",
    )

    return global_oof


def get_outer_global_scores(
    wpt: Any,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
) -> pd.DataFrame:
    model = wpt.create_model(
        BASE_CONFIGURATION
    )

    model.fit(
        X[outer_train_indices],
        y[outer_train_indices],
    )

    test_window_scores = (
        wpt.get_decision_scores(
            fitted_model=model,
            X=X[outer_test_indices],
        )
    )

    subject_scores = (
        wpt.aggregate_subject_scores(
            metadata=metadata.iloc[
                outer_test_indices
            ],
            y_true=y[outer_test_indices],
            scores=test_window_scores,
        )
    )

    selected_features = (
        wpt.get_selected_feature_names(
            fitted_model=model,
            feature_columns=(
                activity_module_feature_columns
            ),
        )
    )

    del model
    gc.collect()

    return subject_scores, selected_features


# Set after loading the dataset.
activity_module_feature_columns: list[str] = []


# ============================================================
# Blend selection
# ============================================================

def select_hybrid_blend(
    wpt: Any,
    merged_inner: pd.DataFrame,
    global_mean: float,
    global_std: float,
    activity_mean: float,
    activity_std: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    y_true = merged_inner[
        "y_true"
    ].to_numpy(dtype=np.int8)

    global_z = standardize_scores(
        scores=merged_inner[
            "global_score"
        ].to_numpy(dtype=float),
        mean_value=global_mean,
        std_value=global_std,
    )

    activity_z = standardize_scores(
        scores=merged_inner[
            "activity_score"
        ].to_numpy(dtype=float),
        mean_value=activity_mean,
        std_value=activity_std,
    )

    best_result: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    search_rows: list[dict[str, Any]] = []

    for global_weight in GLOBAL_WEIGHT_CANDIDATES:
        activity_weight = 1.0 - global_weight

        blended_scores = (
            global_weight * global_z
            + activity_weight * activity_z
        )

        threshold, metrics = (
            wpt.find_best_threshold(
                y_true=y_true,
                scores=blended_scores,
            )
        )

        minimum_recall = min(
            metrics["healthy_recall"],
            metrics["parkinson_recall"],
        )

        blend_std = safe_standard_deviation(
            blended_scores
        )

        search_rows.append(
            {
                "global_weight": (
                    float(global_weight)
                ),
                "activity_weight": (
                    float(activity_weight)
                ),
                "selected_threshold": (
                    float(threshold)
                ),
                "blend_score_std": blend_std,
                **metrics,
            }
        )

        selection_key = (
            metrics["macro_f1"],
            minimum_recall,
            metrics["balanced_accuracy"],
            metrics["roc_auc"],
            # Prefer a genuine hybrid only when the metrics tie.
            -abs(float(global_weight) - 0.5),
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_result = {
                "global_weight": (
                    float(global_weight)
                ),
                "activity_weight": (
                    float(activity_weight)
                ),
                "threshold": float(threshold),
                "inner_metrics": metrics,
                "inner_blended_scores": (
                    blended_scores
                ),
                "inner_blend_std": blend_std,
            }

    if best_result is None:
        raise RuntimeError(
            "Hybrid blend selection failed."
        )

    return best_result, search_rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    global activity_module_feature_columns

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    wpt = load_python_module(
        WPT_SCRIPT,
        "wpt_tuning_module",
    )

    activity = load_python_module(
        ACTIVITY_SCRIPT,
        "wpt_activity_module",
    )

    (
        metadata,
        X,
        y,
        groups,
        feature_columns,
    ) = wpt.load_wpt_dataset()

    activity_module_feature_columns = (
        feature_columns
    )

    # Required by build_outer_test_matrix in step 13.
    activity.wpt_feature_columns_global = (
        feature_columns
    )

    activities = sorted(
        metadata["activity"]
        .astype(str)
        .unique()
    )

    if len(activities) != 11:
        raise ValueError(
            f"Expected 11 activities, found {len(activities)}."
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_splits = list(
        outer_cv.split(
            X=np.zeros(
                (len(y), 1),
                dtype=np.float32,
            ),
            y=y,
            groups=groups,
        )
    )

    fold_rows: list[dict[str, Any]] = []
    blend_search_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    activity_selection_rows: list[dict[str, Any]] = []

    print("=" * 82)
    print("NESTED HYBRID GLOBAL-WPT + ACTIVITY-WPT BLEND")
    print("=" * 82)
    print(f"Rows: {len(metadata)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Subjects: {metadata['subject_id'].nunique()}")
    print(f"Activities: {len(activities)}")
    print(f"Base configuration: {BASE_CONFIGURATION}")
    print(
        "Global-weight candidates: "
        + ", ".join(
            f"{value:.1f}"
            for value in GLOBAL_WEIGHT_CANDIDATES
        )
    )

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print("\n" + "=" * 82)
        print(
            f"OUTER FOLD {outer_fold}/{N_OUTER_SPLITS}"
        )
        print("=" * 82)

        train_subjects = set(
            groups[outer_train_indices]
        )

        test_subjects = set(
            groups[outer_test_indices]
        )

        if train_subjects.intersection(
            test_subjects
        ):
            raise RuntimeError(
                "Subject leakage in outer CV."
            )

        print(
            f"Train subjects: {len(train_subjects)}"
        )
        print(
            f"Test subjects: {len(test_subjects)}"
        )

        inner_cv = StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=(
                RANDOM_STATE + outer_fold
            ),
        )

        inner_splits_relative = list(
            inner_cv.split(
                X=np.zeros(
                    (
                        len(outer_train_indices),
                        1,
                    ),
                    dtype=np.float32,
                ),
                y=y[outer_train_indices],
                groups=groups[
                    outer_train_indices
                ],
            )
        )

        # ----------------------------------------------------
        # Inner global stream
        # ----------------------------------------------------

        inner_global = (
            get_inner_global_oof_scores(
                wpt=wpt,
                metadata=metadata,
                X=X,
                y=y,
                groups=groups,
                outer_train_indices=(
                    outer_train_indices
                ),
                inner_splits_relative=(
                    inner_splits_relative
                ),
            )
        )

        inner_global = inner_global[
            [
                "subject_id",
                "y_true",
                "score",
            ]
        ].rename(
            columns={
                "score": "global_score",
            }
        )

        # ----------------------------------------------------
        # Inner activity stream
        # ----------------------------------------------------

        (
            inner_activity_table,
            activity_auc,
            activity_score_mean,
            activity_score_std,
        ) = activity.build_inner_activity_matrix(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            groups=groups,
            outer_train_indices=(
                outer_train_indices
            ),
            inner_splits_relative=(
                inner_splits_relative
            ),
            activities=activities,
        )

        activity_selection = (
            activity.select_activity_subset(
                wpt=wpt,
                subject_table=(
                    inner_activity_table
                ),
                activity_auc=activity_auc,
                activity_mean=(
                    activity_score_mean
                ),
                activity_std=(
                    activity_score_std
                ),
            )
        )

        selected_activities = (
            activity_selection[
                "selected_activities"
            ]
        )

        activity_weights = (
            activity_selection["weights"]
        )

        inner_activity = (
            inner_activity_table[
                ["subject_id", "y_true"]
            ].copy()
        )

        inner_activity[
            "activity_score"
        ] = activity_selection[
            "inner_scores"
        ]

        # ----------------------------------------------------
        # Align streams
        # ----------------------------------------------------

        merged_inner = inner_global.merge(
            inner_activity,
            on="subject_id",
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_global",
                "_activity",
            ),
        )

        if not np.array_equal(
            merged_inner[
                "y_true_global"
            ].to_numpy(dtype=np.int8),
            merged_inner[
                "y_true_activity"
            ].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(
                "Global and activity labels do not match."
            )

        merged_inner = merged_inner.rename(
            columns={
                "y_true_global": "y_true",
            }
        ).drop(
            columns=["y_true_activity"]
        )

        assert_subject_table(
            merged_inner,
            train_subjects,
            "merged inner stream scores",
        )

        global_mean = float(
            merged_inner[
                "global_score"
            ].mean()
        )

        global_std = safe_standard_deviation(
            merged_inner[
                "global_score"
            ].to_numpy(dtype=float)
        )

        combined_activity_mean = float(
            merged_inner[
                "activity_score"
            ].mean()
        )

        combined_activity_std = (
            safe_standard_deviation(
                merged_inner[
                    "activity_score"
                ].to_numpy(dtype=float)
            )
        )

        blend_selection, current_search = (
            select_hybrid_blend(
                wpt=wpt,
                merged_inner=merged_inner,
                global_mean=global_mean,
                global_std=global_std,
                activity_mean=(
                    combined_activity_mean
                ),
                activity_std=(
                    combined_activity_std
                ),
            )
        )

        for row in current_search:
            blend_search_rows.append(
                {
                    "outer_fold": outer_fold,
                    "selected_activity_count": (
                        len(selected_activities)
                    ),
                    "selected_activities": (
                        "|".join(
                            selected_activities
                        )
                    ),
                    **row,
                }
            )

        selected_global_weight = (
            blend_selection[
                "global_weight"
            ]
        )

        selected_activity_weight = (
            blend_selection[
                "activity_weight"
            ]
        )

        selected_threshold = (
            blend_selection["threshold"]
        )

        inner_blend_std = (
            blend_selection[
                "inner_blend_std"
            ]
        )

        inner_metrics = (
            blend_selection[
                "inner_metrics"
            ]
        )

        ranked_activities = sorted(
            activities,
            key=activity_auc.get,
            reverse=True,
        )

        for rank, activity_name in enumerate(
            ranked_activities,
            start=1,
        ):
            activity_selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "activity": activity_name,
                    "inner_rank": rank,
                    "inner_roc_auc": (
                        activity_auc[
                            activity_name
                        ]
                    ),
                    "selected": (
                        activity_name
                        in selected_activities
                    ),
                    "activity_weight_inside_stream": (
                        activity_weights.get(
                            activity_name,
                            0.0,
                        )
                    ),
                    "selected_global_stream_weight": (
                        selected_global_weight
                    ),
                    "selected_activity_stream_weight": (
                        selected_activity_weight
                    ),
                }
            )

        print(
            "Selected activities: "
            + ", ".join(selected_activities)
        )
        print(
            "Selected activity count: "
            f"{len(selected_activities)}"
        )
        print(
            "Selected stream weights: "
            f"global={selected_global_weight:.1f}, "
            f"activity={selected_activity_weight:.1f}"
        )
        print(
            "Selected threshold: "
            f"{selected_threshold:.6f}"
        )
        print(
            "Inner Macro-F1: "
            f"{inner_metrics['macro_f1']:.4f}"
        )
        print(
            "Inner recalls: "
            f"Healthy={inner_metrics['healthy_recall']:.4f}, "
            f"Parkinson={inner_metrics['parkinson_recall']:.4f}"
        )

        # ----------------------------------------------------
        # Outer-test global stream
        # ----------------------------------------------------

        (
            outer_global,
            global_selected_features,
        ) = get_outer_global_scores(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            outer_train_indices=(
                outer_train_indices
            ),
            outer_test_indices=(
                outer_test_indices
            ),
        )

        outer_global = outer_global[
            [
                "subject_id",
                "y_true",
                "score",
            ]
        ].rename(
            columns={
                "score": "global_score",
            }
        )

        # ----------------------------------------------------
        # Outer-test activity stream
        # ----------------------------------------------------

        (
            outer_activity_table,
            selected_features_by_activity,
        ) = activity.build_outer_test_matrix(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            outer_train_indices=(
                outer_train_indices
            ),
            outer_test_indices=(
                outer_test_indices
            ),
            selected_activities=(
                selected_activities
            ),
        )

        outer_activity_scores = (
            activity.calculate_combined_scores(
                subject_table=(
                    outer_activity_table
                ),
                selected_activities=(
                    selected_activities
                ),
                weights=activity_weights,
                activity_mean=(
                    activity_score_mean
                ),
                activity_std=(
                    activity_score_std
                ),
            )
        )

        outer_activity = (
            outer_activity_table[
                ["subject_id", "y_true"]
            ].copy()
        )

        outer_activity[
            "activity_score"
        ] = outer_activity_scores

        merged_outer = outer_global.merge(
            outer_activity,
            on="subject_id",
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_global",
                "_activity",
            ),
        )

        if not np.array_equal(
            merged_outer[
                "y_true_global"
            ].to_numpy(dtype=np.int8),
            merged_outer[
                "y_true_activity"
            ].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(
                "Outer global and activity labels do not match."
            )

        merged_outer = merged_outer.rename(
            columns={
                "y_true_global": "y_true",
            }
        ).drop(
            columns=["y_true_activity"]
        )

        assert_subject_table(
            merged_outer,
            test_subjects,
            "merged outer stream scores",
        )

        outer_global_z = standardize_scores(
            scores=merged_outer[
                "global_score"
            ].to_numpy(dtype=float),
            mean_value=global_mean,
            std_value=global_std,
        )

        outer_activity_z = standardize_scores(
            scores=merged_outer[
                "activity_score"
            ].to_numpy(dtype=float),
            mean_value=(
                combined_activity_mean
            ),
            std_value=(
                combined_activity_std
            ),
        )

        outer_blended_scores = (
            selected_global_weight
            * outer_global_z
            + selected_activity_weight
            * outer_activity_z
        )

        outer_y_true = merged_outer[
            "y_true"
        ].to_numpy(dtype=np.int8)

        outer_y_pred = (
            outer_blended_scores
            >= selected_threshold
        ).astype(np.int8)

        outer_metrics = wpt.calculate_metrics(
            y_true=outer_y_true,
            y_pred=outer_y_pred,
            scores=outer_blended_scores,
        )

        normalized_margin = (
            outer_blended_scores
            - selected_threshold
        ) / inner_blend_std

        inner_stream_correlation = float(
            np.corrcoef(
                merged_inner[
                    "global_score"
                ].to_numpy(dtype=float),
                merged_inner[
                    "activity_score"
                ].to_numpy(dtype=float),
            )[0, 1]
        )

        outer_stream_correlation = float(
            np.corrcoef(
                merged_outer[
                    "global_score"
                ].to_numpy(dtype=float),
                merged_outer[
                    "activity_score"
                ].to_numpy(dtype=float),
            )[0, 1]
        )

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "train_subjects": len(
                    train_subjects
                ),
                "test_subjects": len(
                    test_subjects
                ),
                "selected_activity_count": (
                    len(selected_activities)
                ),
                "selected_activities": (
                    "|".join(
                        selected_activities
                    )
                ),
                "global_weight": (
                    selected_global_weight
                ),
                "activity_weight": (
                    selected_activity_weight
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "inner_blend_std": (
                    inner_blend_std
                ),
                "inner_stream_correlation": (
                    inner_stream_correlation
                ),
                "outer_stream_correlation": (
                    outer_stream_correlation
                ),
                "global_selected_feature_count": (
                    len(global_selected_features)
                ),
                "activity_model_count": (
                    len(
                        selected_features_by_activity
                    )
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

        prediction_frame = merged_outer[
            ["subject_id", "y_true"]
        ].copy()

        prediction_frame[
            "outer_fold"
        ] = outer_fold

        prediction_frame[
            "global_score"
        ] = merged_outer[
            "global_score"
        ].to_numpy(dtype=float)

        prediction_frame[
            "activity_score"
        ] = merged_outer[
            "activity_score"
        ].to_numpy(dtype=float)

        prediction_frame[
            "global_z"
        ] = outer_global_z

        prediction_frame[
            "activity_z"
        ] = outer_activity_z

        prediction_frame[
            "blended_score"
        ] = outer_blended_scores

        prediction_frame[
            "normalized_margin"
        ] = normalized_margin

        prediction_frame[
            "selected_threshold"
        ] = selected_threshold

        prediction_frame[
            "global_weight"
        ] = selected_global_weight

        prediction_frame[
            "activity_weight"
        ] = selected_activity_weight

        prediction_frame[
            "y_pred"
        ] = outer_y_pred

        prediction_frame[
            "true_label"
        ] = prediction_frame[
            "y_true"
        ].map(ID_TO_LABEL)

        prediction_frame[
            "predicted_label"
        ] = prediction_frame[
            "y_pred"
        ].map(ID_TO_LABEL)

        prediction_frame[
            "selected_activities"
        ] = "|".join(
            selected_activities
        )

        prediction_frames.append(
            prediction_frame
        )

        print("\nOuter-test results:")
        print(
            "Accuracy:         "
            f"{outer_metrics['accuracy']:.4f}"
        )
        print(
            "Balanced Acc:     "
            f"{outer_metrics['balanced_accuracy']:.4f}"
        )
        print(
            "Macro-F1:         "
            f"{outer_metrics['macro_f1']:.4f}"
        )
        print(
            "Healthy Recall:   "
            f"{outer_metrics['healthy_recall']:.4f}"
        )
        print(
            "Parkinson Recall: "
            f"{outer_metrics['parkinson_recall']:.4f}"
        )
        print(
            "ROC-AUC:          "
            f"{outer_metrics['roc_auc']:.4f}"
        )
        print(
            "Stream correlation: "
            f"inner={inner_stream_correlation:.4f}, "
            f"outer={outer_stream_correlation:.4f}"
        )

        gc.collect()

    # ========================================================
    # Overall results
    # ========================================================

    fold_table = pd.DataFrame(
        fold_rows
    )

    blend_search_table = pd.DataFrame(
        blend_search_rows
    )

    activity_selection_table = (
        pd.DataFrame(
            activity_selection_rows
        )
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if len(predictions) != 355:
        raise RuntimeError(
            "Expected exactly 355 subject predictions."
        )

    if predictions[
        "subject_id"
    ].nunique() != 355:
        raise RuntimeError(
            "Subjects are duplicated or missing."
        )

    y_true = predictions[
        "y_true"
    ].to_numpy(dtype=np.int8)

    y_pred = predictions[
        "y_pred"
    ].to_numpy(dtype=np.int8)

    normalized_scores = predictions[
        "normalized_margin"
    ].to_numpy(dtype=float)

    overall_metrics = wpt.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=normalized_scores,
    )

    summary = pd.DataFrame(
        [
            {
                "method": (
                    "nested_global_wpt_activity_wpt_blend"
                ),
                "n_subjects": len(predictions),
                **overall_metrics,
                "fold_accuracy_mean": float(
                    fold_table[
                        "outer_accuracy"
                    ].mean()
                ),
                "fold_accuracy_std": float(
                    fold_table[
                        "outer_accuracy"
                    ].std()
                ),
                "fold_balanced_accuracy_mean": float(
                    fold_table[
                        "outer_balanced_accuracy"
                    ].mean()
                ),
                "fold_balanced_accuracy_std": float(
                    fold_table[
                        "outer_balanced_accuracy"
                    ].std()
                ),
                "fold_macro_f1_mean": float(
                    fold_table[
                        "outer_macro_f1"
                    ].mean()
                ),
                "fold_macro_f1_std": float(
                    fold_table[
                        "outer_macro_f1"
                    ].std()
                ),
                "fold_roc_auc_mean": float(
                    fold_table[
                        "outer_roc_auc"
                    ].mean()
                ),
                "fold_roc_auc_std": float(
                    fold_table[
                        "outer_roc_auc"
                    ].std()
                ),
                "mean_global_weight": float(
                    fold_table[
                        "global_weight"
                    ].mean()
                ),
                "mean_activity_weight": float(
                    fold_table[
                        "activity_weight"
                    ].mean()
                ),
            }
        ]
    )

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
        "Nested hybrid global/activity WPT blend"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

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

    CLASSIFICATION_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    blend_search_table.to_csv(
        BLEND_SEARCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    activity_selection_table.to_csv(
        ACTIVITY_SELECTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 88)
    print("OVERALL HYBRID WPT BLEND RESULTS")
    print("=" * 88)

    print(
        summary[
            [
                "method",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "healthy_recall",
                "parkinson_recall",
                "roc_auc",
                "fold_roc_auc_mean",
                "mean_global_weight",
                "mean_activity_weight",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nSelected stream weights by fold:")

    print(
        fold_table[
            [
                "outer_fold",
                "global_weight",
                "activity_weight",
                "selected_activity_count",
                "selected_threshold",
                "outer_accuracy",
                "outer_balanced_accuracy",
                "outer_macro_f1",
                "outer_roc_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nCreated files:")

    for output_path in [
        FOLD_RESULTS_PATH,
        BLEND_SEARCH_PATH,
        SUBJECT_PREDICTIONS_PATH,
        ACTIVITY_SELECTION_PATH,
        OVERALL_SUMMARY_PATH,
        CONFUSION_MATRIX_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
