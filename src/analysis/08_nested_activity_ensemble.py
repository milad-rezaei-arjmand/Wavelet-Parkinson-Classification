from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
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


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_SCRIPT = (
    PROJECT_ROOT / "05_train_baselines.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "nested_activity_ensemble"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR / "fold_results.csv"
)

SUBJECT_PREDICTIONS_PATH = (
    OUTPUT_DIR / "subject_predictions.csv"
)

ACTIVITY_SELECTION_PATH = (
    OUTPUT_DIR / "activity_selection.csv"
)

OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR / "overall_summary.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "subject_confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR / "classification_report.txt"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

TOP_ACTIVITY_COUNTS = [
    3,
    5,
    7,
    9,
    11,
]

EPSILON = 1e-12

ID_TO_LABEL = {
    0: "Healthy",
    1: "Parkinson",
}


# ============================================================
# Load previous code
# ============================================================

def load_baseline_module():
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Baseline script not found: {BASELINE_SCRIPT}"
        )

    specification = importlib.util.spec_from_file_location(
        "baseline_module",
        BASELINE_SCRIPT,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load baseline script."
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
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


# ============================================================
# Activity-level aggregation
# ============================================================

def aggregate_activity_scores(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:

    temporary = metadata[
        ["subject_id"]
    ].copy()

    temporary["y_true"] = y_true
    temporary["score"] = scores

    label_counts = (
        temporary.groupby("subject_id")[
            "y_true"
        ].nunique()
    )

    if not np.all(label_counts == 1):
        raise ValueError(
            "Inconsistent labels inside an activity."
        )

    aggregated = (
        temporary.groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    return aggregated


# ============================================================
# Threshold selection
# ============================================================

def create_threshold_candidates(
    scores: np.ndarray,
) -> np.ndarray:

    unique_scores = np.unique(
        np.asarray(scores, dtype=float)
    )

    if unique_scores.size == 1:
        return np.array(
            [
                unique_scores[0] - 1e-6,
                unique_scores[0],
                unique_scores[0] + 1e-6,
            ]
        )

    middle_points = (
        unique_scores[:-1]
        + unique_scores[1:]
    ) / 2

    return np.concatenate(
        [
            [unique_scores[0] - 1e-6],
            middle_points,
            [unique_scores[-1] + 1e-6],
        ]
    )


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:

    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in create_threshold_candidates(
        scores
    ):
        y_pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
        )

        minimum_recall = min(
            metrics["healthy_recall"],
            metrics["parkinson_recall"],
        )

        selection_key = (
            metrics["macro_f1"],
            minimum_recall,
            metrics["balanced_accuracy"],
            metrics["accuracy"],
        )

        if (
            best_key is None
            or selection_key > best_key
        ):
            best_key = selection_key
            best_threshold = float(threshold)
            best_metrics = metrics

    if (
        best_threshold is None
        or best_metrics is None
    ):
        raise RuntimeError(
            "Threshold selection failed."
        )

    return best_threshold, best_metrics


# ============================================================
# Inner OOF scores for one activity
# ============================================================

def get_activity_inner_oof_scores(
    baseline: Any,
    model_template: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    activity: str,
    outer_fold: int,
) -> pd.DataFrame:

    activity_mask = (
        data.iloc[outer_train_indices][
            "activity"
        ].to_numpy()
        == activity
    )

    activity_indices = (
        outer_train_indices[activity_mask]
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=(
            RANDOM_STATE
            + outer_fold
            + sum(ord(char) for char in activity)
        ),
    )

    activity_frames: list[pd.DataFrame] = []

    inner_splits = inner_cv.split(
        X=X[activity_indices],
        y=y[activity_indices],
        groups=groups[activity_indices],
    )

    for inner_fold, (
        train_relative,
        validation_relative,
    ) in enumerate(
        inner_splits,
        start=1,
    ):
        train_indices = activity_indices[
            train_relative
        ]

        validation_indices = activity_indices[
            validation_relative
        ]

        train_subjects = set(
            groups[train_indices]
        )

        validation_subjects = set(
            groups[validation_indices]
        )

        if train_subjects.intersection(
            validation_subjects
        ):
            raise RuntimeError(
                "Subject leakage in inner CV."
            )

        model = clone(model_template)

        model.fit(
            X[train_indices],
            y[train_indices],
        )

        validation_scores, _ = (
            baseline.get_prediction_scores(
                model,
                X[validation_indices],
            )
        )

        subject_scores = aggregate_activity_scores(
            metadata=data.iloc[
                validation_indices
            ],
            y_true=y[validation_indices],
            scores=validation_scores,
        )

        subject_scores["inner_fold"] = (
            inner_fold
        )

        activity_frames.append(
            subject_scores
        )

    activity_oof = pd.concat(
        activity_frames,
        ignore_index=True,
    )

    expected_subjects = set(
        groups[outer_train_indices]
    )

    predicted_subjects = set(
        activity_oof["subject_id"]
    )

    if expected_subjects != predicted_subjects:
        raise RuntimeError(
            f"Invalid OOF subject set for "
            f"activity {activity}."
        )

    if (
        activity_oof["subject_id"]
        .value_counts()
        .ne(1)
        .any()
    ):
        raise RuntimeError(
            f"Duplicated OOF subjects for "
            f"activity {activity}."
        )

    return activity_oof


# ============================================================
# Build inner subject/activity matrix
# ============================================================

def build_inner_activity_matrix(
    baseline: Any,
    model_template: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    activities: list[str],
    outer_fold: int,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:

    subject_table: pd.DataFrame | None = None

    activity_auc: dict[str, float] = {}
    activity_mean: dict[str, float] = {}
    activity_std: dict[str, float] = {}

    for activity in activities:

        activity_scores = (
            get_activity_inner_oof_scores(
                baseline=baseline,
                model_template=model_template,
                data=data,
                X=X,
                y=y,
                groups=groups,
                outer_train_indices=(
                    outer_train_indices
                ),
                activity=activity,
                outer_fold=outer_fold,
            )
        )

        activity_scores = (
            activity_scores[
                [
                    "subject_id",
                    "y_true",
                    "score",
                ]
            ]
            .rename(
                columns={
                    "score": (
                        f"score__{activity}"
                    )
                }
            )
        )

        score_column = (
            f"score__{activity}"
        )

        if subject_table is None:
            subject_table = activity_scores
        else:
            subject_table = subject_table.merge(
                activity_scores[
                    [
                        "subject_id",
                        score_column,
                    ]
                ],
                on="subject_id",
                how="inner",
                validate="one_to_one",
            )

        scores = activity_scores[
            score_column
        ].to_numpy(dtype=float)

        labels = activity_scores[
            "y_true"
        ].to_numpy(dtype=np.int8)

        activity_auc[activity] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        activity_mean[activity] = float(
            np.mean(scores)
        )

        activity_std[activity] = float(
            np.std(scores)
        )

    if subject_table is None:
        raise RuntimeError(
            "No activity matrix was created."
        )

    expected_subject_count = len(
        set(groups[outer_train_indices])
    )

    if len(subject_table) != expected_subject_count:
        raise RuntimeError(
            "Inner activity matrix has an "
            "unexpected subject count."
        )

    return (
        subject_table,
        activity_auc,
        activity_mean,
        activity_std,
    )


# ============================================================
# Activity weighting and subset selection
# ============================================================

def calculate_activity_weights(
    selected_activities: list[str],
    activity_auc: dict[str, float],
) -> dict[str, float]:

    raw_weights = {
        activity: max(
            activity_auc[activity] - 0.5,
            0.01,
        ) ** 2
        for activity in selected_activities
    }

    total_weight = sum(
        raw_weights.values()
    )

    return {
        activity: (
            raw_weights[activity]
            / total_weight
        )
        for activity in selected_activities
    }


def calculate_combined_scores(
    subject_table: pd.DataFrame,
    selected_activities: list[str],
    weights: dict[str, float],
    activity_mean: dict[str, float],
    activity_std: dict[str, float],
) -> np.ndarray:

    combined_scores = np.zeros(
        len(subject_table),
        dtype=float,
    )

    for activity in selected_activities:

        scores = subject_table[
            f"score__{activity}"
        ].to_numpy(dtype=float)

        standard_deviation = (
            activity_std[activity]
        )

        if standard_deviation <= EPSILON:
            standardized_scores = (
                scores - activity_mean[activity]
            )
        else:
            standardized_scores = (
                scores - activity_mean[activity]
            ) / standard_deviation

        combined_scores += (
            weights[activity]
            * standardized_scores
        )

    return combined_scores


def select_activity_subset(
    subject_table: pd.DataFrame,
    activity_auc: dict[str, float],
    activity_mean: dict[str, float],
    activity_std: dict[str, float],
) -> dict[str, Any]:

    ranked_activities = sorted(
        activity_auc,
        key=activity_auc.get,
        reverse=True,
    )

    y_true = subject_table[
        "y_true"
    ].to_numpy(dtype=np.int8)

    best_result: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    maximum_activity_count = len(
        ranked_activities
    )

    candidate_counts = [
        count
        for count in TOP_ACTIVITY_COUNTS
        if count <= maximum_activity_count
    ]

    if maximum_activity_count not in candidate_counts:
        candidate_counts.append(
            maximum_activity_count
        )

    for activity_count in sorted(
        set(candidate_counts)
    ):

        selected_activities = (
            ranked_activities[
                :activity_count
            ]
        )

        weights = calculate_activity_weights(
            selected_activities=(
                selected_activities
            ),
            activity_auc=activity_auc,
        )

        combined_scores = (
            calculate_combined_scores(
                subject_table=subject_table,
                selected_activities=(
                    selected_activities
                ),
                weights=weights,
                activity_mean=activity_mean,
                activity_std=activity_std,
            )
        )

        threshold, metrics = (
            find_best_threshold(
                y_true=y_true,
                scores=combined_scores,
            )
        )

        minimum_recall = min(
            metrics["healthy_recall"],
            metrics["parkinson_recall"],
        )

        selection_key = (
            metrics["macro_f1"],
            minimum_recall,
            metrics["balanced_accuracy"],
            metrics["roc_auc"],
            -activity_count,
        )

        if (
            best_key is None
            or selection_key > best_key
        ):
            best_key = selection_key

            best_result = {
                "selected_activities": (
                    selected_activities
                ),
                "activity_count": (
                    activity_count
                ),
                "weights": weights,
                "threshold": threshold,
                "inner_metrics": metrics,
                "inner_scores": combined_scores,
            }

    if best_result is None:
        raise RuntimeError(
            "Activity subset selection failed."
        )

    return best_result


# ============================================================
# Outer-test activity scores
# ============================================================

def get_outer_test_activity_scores(
    baseline: Any,
    model_template: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
    activity: str,
) -> pd.DataFrame:

    train_mask = (
        data.iloc[outer_train_indices][
            "activity"
        ].to_numpy()
        == activity
    )

    test_mask = (
        data.iloc[outer_test_indices][
            "activity"
        ].to_numpy()
        == activity
    )

    train_indices = (
        outer_train_indices[train_mask]
    )

    test_indices = (
        outer_test_indices[test_mask]
    )

    model = clone(model_template)

    model.fit(
        X[train_indices],
        y[train_indices],
    )

    test_scores, _ = (
        baseline.get_prediction_scores(
            model,
            X[test_indices],
        )
    )

    subject_scores = aggregate_activity_scores(
        metadata=data.iloc[test_indices],
        y_true=y[test_indices],
        scores=test_scores,
    )

    return (
        subject_scores[
            [
                "subject_id",
                "y_true",
                "score",
            ]
        ]
        .rename(
            columns={
                "score": f"score__{activity}"
            }
        )
    )


def build_outer_test_matrix(
    baseline: Any,
    model_template: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
    selected_activities: list[str],
) -> pd.DataFrame:

    subject_table: pd.DataFrame | None = None

    for activity in selected_activities:

        activity_scores = (
            get_outer_test_activity_scores(
                baseline=baseline,
                model_template=model_template,
                data=data,
                X=X,
                y=y,
                outer_train_indices=(
                    outer_train_indices
                ),
                outer_test_indices=(
                    outer_test_indices
                ),
                activity=activity,
            )
        )

        score_column = (
            f"score__{activity}"
        )

        if subject_table is None:
            subject_table = activity_scores
        else:
            subject_table = subject_table.merge(
                activity_scores[
                    [
                        "subject_id",
                        score_column,
                    ]
                ],
                on="subject_id",
                how="inner",
                validate="one_to_one",
            )

    if subject_table is None:
        raise RuntimeError(
            "No outer-test activity matrix created."
        )

    return subject_table


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_baseline_module()

    (
        data,
        X,
        y,
        groups,
        feature_columns,
    ) = baseline.load_dataset()

    model_template = (
        baseline.create_models()["SVM_RBF"]
    )

    activities = sorted(
        data["activity"].unique()
    )

    if len(activities) != 11:
        raise ValueError(
            f"Expected 11 activities, "
            f"found {len(activities)}."
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    activity_selection_rows: list[
        dict[str, Any]
    ] = []

    print("=" * 75)
    print("NESTED ACTIVITY-WEIGHTED SVM ENSEMBLE")
    print("=" * 75)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Subjects: "
        f"{data['subject_id'].nunique()}"
    )
    print(f"Activities: {len(activities)}")

    outer_splits = outer_cv.split(
        X=X,
        y=y,
        groups=groups,
    )

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        print("\n" + "=" * 75)
        print(
            f"OUTER FOLD "
            f"{outer_fold}/{N_OUTER_SPLITS}"
        )
        print("=" * 75)

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
                "Subject leakage in outer fold."
            )

        print(
            f"Train subjects: "
            f"{len(train_subjects)}"
        )

        print(
            f"Test subjects: "
            f"{len(test_subjects)}"
        )

        (
            inner_subject_table,
            activity_auc,
            activity_mean,
            activity_std,
        ) = build_inner_activity_matrix(
            baseline=baseline,
            model_template=model_template,
            data=data,
            X=X,
            y=y,
            groups=groups,
            outer_train_indices=(
                outer_train_indices
            ),
            activities=activities,
            outer_fold=outer_fold,
        )

        selection = select_activity_subset(
            subject_table=inner_subject_table,
            activity_auc=activity_auc,
            activity_mean=activity_mean,
            activity_std=activity_std,
        )

        selected_activities = selection[
            "selected_activities"
        ]

        weights = selection["weights"]
        threshold = selection["threshold"]

        inner_metrics = selection[
            "inner_metrics"
        ]

        print(
            "Selected activity count: "
            f"{len(selected_activities)}"
        )

        print(
            "Selected activities: "
            + ", ".join(selected_activities)
        )

        print(
            "Inner Macro-F1: "
            f"{inner_metrics['macro_f1']:.4f}"
        )

        print(
            "Inner Healthy Recall: "
            f"{inner_metrics['healthy_recall']:.4f}"
        )

        print(
            "Inner Parkinson Recall: "
            f"{inner_metrics['parkinson_recall']:.4f}"
        )

        print(
            "Selected threshold: "
            f"{threshold:.6f}"
        )

        for rank, activity in enumerate(
            sorted(
                activities,
                key=activity_auc.get,
                reverse=True,
            ),
            start=1,
        ):
            activity_selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "activity": activity,
                    "inner_rank": rank,
                    "inner_roc_auc": (
                        activity_auc[activity]
                    ),
                    "selected": (
                        activity
                        in selected_activities
                    ),
                    "weight": (
                        weights.get(
                            activity,
                            0.0,
                        )
                    ),
                    "selected_activity_count": (
                        len(selected_activities)
                    ),
                    "selected_threshold": (
                        threshold
                    ),
                }
            )

        outer_subject_table = (
            build_outer_test_matrix(
                baseline=baseline,
                model_template=model_template,
                data=data,
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
        )

        outer_scores = calculate_combined_scores(
            subject_table=outer_subject_table,
            selected_activities=(
                selected_activities
            ),
            weights=weights,
            activity_mean=activity_mean,
            activity_std=activity_std,
        )

        outer_y_true = outer_subject_table[
            "y_true"
        ].to_numpy(dtype=np.int8)

        outer_y_pred = (
            outer_scores >= threshold
        ).astype(np.int8)

        outer_metrics = calculate_metrics(
            y_true=outer_y_true,
            y_pred=outer_y_pred,
            scores=outer_scores,
        )

        prediction_frame = (
            outer_subject_table[
                [
                    "subject_id",
                    "y_true",
                ]
            ].copy()
        )

        prediction_frame[
            "outer_fold"
        ] = outer_fold

        prediction_frame[
            "score"
        ] = outer_scores

        prediction_frame[
            "threshold"
        ] = threshold

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
                "selected_threshold": threshold,
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

    # ========================================================
    # Overall evaluation
    # ========================================================

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if len(predictions) != 355:
        raise RuntimeError(
            "Expected 355 subject predictions, "
            f"found {len(predictions)}."
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

    scores = predictions[
        "score"
    ].to_numpy(dtype=float)

    overall_metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
    )

    overall_summary = pd.DataFrame(
        [
            {
                "method": (
                    "nested_activity_weighted_svm"
                ),
                "n_subjects": len(predictions),
                **overall_metrics,
            }
        ]
    )

    fold_table = pd.DataFrame(
        fold_rows
    )

    activity_selection_table = (
        pd.DataFrame(
            activity_selection_rows
        )
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

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
        "Nested activity-weighted SVM"
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

    # ========================================================
    # Save results
    # ========================================================

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
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

    overall_summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n" + "=" * 75)
    print("OVERALL OUTER-FOLD RESULTS")
    print("=" * 75)

    print(
        overall_summary[
            [
                "method",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "healthy_recall",
                "parkinson_recall",
                "roc_auc",
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

    print("\nActivity selection frequency:")

    selection_frequency = (
        activity_selection_table.groupby(
            "activity"
        )
        .agg(
            selected_folds=(
                "selected",
                "sum",
            ),
            mean_inner_rank=(
                "inner_rank",
                "mean",
            ),
            mean_inner_auc=(
                "inner_roc_auc",
                "mean",
            ),
            mean_weight=(
                "weight",
                "mean",
            ),
        )
        .sort_values(
            [
                "selected_folds",
                "mean_inner_auc",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print(
        selection_frequency.to_string(
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    print("\nCreated files:")

    for path in [
        FOLD_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        ACTIVITY_SELECTION_PATH,
        OVERALL_SUMMARY_PATH,
        CONFUSION_MATRIX_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()