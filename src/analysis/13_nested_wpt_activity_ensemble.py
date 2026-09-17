from __future__ import annotations

import gc
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
WPT_SCRIPT = PROJECT_ROOT / "12_tune_wpt_svm.py"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "nested_wpt_activity_ensemble"
)

FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
ACTIVITY_SELECTION_PATH = OUTPUT_DIR / "activity_selection.csv"
SUBJECT_PREDICTIONS_PATH = OUTPUT_DIR / "subject_predictions.csv"
OVERALL_SUMMARY_PATH = OUTPUT_DIR / "overall_summary.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "selected_feature_frequency.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "subject_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

# Fixed after the honest WPT representation screening.
# We intentionally do not reuse the fold-specific tuned settings from step 12,
# because they reduced the overall balanced accuracy and Macro-F1.
BASE_CONFIGURATION = {
    "selected_features": 300,
    "C": 10.0,
    "gamma": "scale",
    "configuration_id": 0,
}

TOP_ACTIVITY_COUNTS = [3, 5, 7, 9, 11]
EPSILON = 1e-12

ID_TO_LABEL = {
    0: "Healthy",
    1: "Parkinson",
}


# ============================================================
# Module loading
# ============================================================

def load_python_module(file_path: Path, module_name: str):
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
# Activity-specific subject aggregation
# ============================================================

def aggregate_activity_scores(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Average all windows of one activity for each subject."""

    temporary = metadata[["subject_id"]].copy()
    temporary["y_true"] = np.asarray(y_true, dtype=np.int8)
    temporary["score"] = np.asarray(scores, dtype=float)

    label_counts = temporary.groupby("subject_id")["y_true"].nunique()

    if not np.all(label_counts == 1):
        raise ValueError(
            "Inconsistent labels inside an activity/subject group."
        )

    aggregated = (
        temporary.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    return aggregated


def activity_indices(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    activity: str,
) -> np.ndarray:
    mask = (
        metadata.iloc[indices]["activity"]
        .astype(str)
        .to_numpy()
        == activity
    )
    return np.asarray(indices)[mask]


# ============================================================
# Inner OOF scores
# ============================================================

def get_inner_oof_scores_for_activity(
    wpt: Any,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    inner_splits_relative: list[tuple[np.ndarray, np.ndarray]],
    activity: str,
) -> pd.DataFrame:

    frames: list[pd.DataFrame] = []

    for inner_fold, (
        inner_train_relative,
        inner_validation_relative,
    ) in enumerate(inner_splits_relative, start=1):

        all_inner_train = outer_train_indices[inner_train_relative]
        all_inner_validation = outer_train_indices[inner_validation_relative]

        train_indices = activity_indices(
            metadata,
            all_inner_train,
            activity,
        )

        validation_indices = activity_indices(
            metadata,
            all_inner_validation,
            activity,
        )

        train_subjects = set(groups[train_indices])
        validation_subjects = set(groups[validation_indices])

        if train_subjects.intersection(validation_subjects):
            raise RuntimeError(
                "Subject leakage detected in inner activity CV."
            )

        model = wpt.create_model(BASE_CONFIGURATION)
        model.fit(X[train_indices], y[train_indices])

        validation_scores = wpt.get_decision_scores(
            fitted_model=model,
            X=X[validation_indices],
        )

        subject_scores = aggregate_activity_scores(
            metadata=metadata.iloc[validation_indices],
            y_true=y[validation_indices],
            scores=validation_scores,
        )

        subject_scores["inner_fold"] = inner_fold
        frames.append(subject_scores)

        del model
        gc.collect()

    activity_oof = pd.concat(frames, ignore_index=True)

    expected_subjects = set(groups[outer_train_indices])
    actual_subjects = set(activity_oof["subject_id"])

    if actual_subjects != expected_subjects:
        missing = sorted(expected_subjects - actual_subjects)
        extra = sorted(actual_subjects - expected_subjects)
        raise RuntimeError(
            f"Invalid OOF subject set for {activity}. "
            f"Missing={missing}, extra={extra}"
        )

    counts = activity_oof["subject_id"].value_counts()

    if not np.all(counts == 1):
        raise RuntimeError(
            f"Duplicated OOF subjects for activity {activity}."
        )

    return activity_oof


def build_inner_activity_matrix(
    wpt: Any,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    inner_splits_relative: list[tuple[np.ndarray, np.ndarray]],
    activities: list[str],
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
        activity_scores = get_inner_oof_scores_for_activity(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            groups=groups,
            outer_train_indices=outer_train_indices,
            inner_splits_relative=inner_splits_relative,
            activity=activity,
        )

        score_column = f"score__{activity}"

        activity_scores = activity_scores[
            ["subject_id", "y_true", "score"]
        ].rename(columns={"score": score_column})

        if subject_table is None:
            subject_table = activity_scores.copy()
        else:
            subject_table = subject_table.merge(
                activity_scores[["subject_id", score_column]],
                on="subject_id",
                how="inner",
                validate="one_to_one",
            )

        scores = activity_scores[score_column].to_numpy(dtype=float)
        labels = activity_scores["y_true"].to_numpy(dtype=np.int8)

        activity_auc[activity] = float(
            roc_auc_score(labels, scores)
        )
        activity_mean[activity] = float(np.mean(scores))
        activity_std[activity] = float(np.std(scores))

    if subject_table is None:
        raise RuntimeError("No inner activity matrix was created.")

    expected_count = len(set(groups[outer_train_indices]))

    if len(subject_table) != expected_count:
        raise RuntimeError(
            "Unexpected inner subject count: "
            f"{len(subject_table)} instead of {expected_count}."
        )

    return subject_table, activity_auc, activity_mean, activity_std


# ============================================================
# Activity combination
# ============================================================

def calculate_activity_weights(
    selected_activities: list[str],
    activity_auc: dict[str, float],
) -> dict[str, float]:

    raw_weights = {
        activity: max(activity_auc[activity] - 0.5, 0.01) ** 2
        for activity in selected_activities
    }

    total_weight = float(sum(raw_weights.values()))

    if total_weight <= EPSILON:
        uniform = 1.0 / len(selected_activities)
        return {
            activity: uniform
            for activity in selected_activities
        }

    return {
        activity: raw_weights[activity] / total_weight
        for activity in selected_activities
    }


def calculate_combined_scores(
    subject_table: pd.DataFrame,
    selected_activities: list[str],
    weights: dict[str, float],
    activity_mean: dict[str, float],
    activity_std: dict[str, float],
) -> np.ndarray:

    combined_scores = np.zeros(len(subject_table), dtype=float)

    for activity in selected_activities:
        scores = subject_table[
            f"score__{activity}"
        ].to_numpy(dtype=float)

        standard_deviation = activity_std[activity]

        if standard_deviation <= EPSILON:
            standardized_scores = scores - activity_mean[activity]
        else:
            standardized_scores = (
                scores - activity_mean[activity]
            ) / standard_deviation

        combined_scores += (
            weights[activity] * standardized_scores
        )

    return combined_scores


def select_activity_subset(
    wpt: Any,
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

    y_true = subject_table["y_true"].to_numpy(dtype=np.int8)

    best_result: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    candidate_counts = sorted(
        {
            count
            for count in TOP_ACTIVITY_COUNTS
            if count <= len(ranked_activities)
        }
        | {len(ranked_activities)}
    )

    for activity_count in candidate_counts:
        selected_activities = ranked_activities[:activity_count]

        weights = calculate_activity_weights(
            selected_activities=selected_activities,
            activity_auc=activity_auc,
        )

        combined_scores = calculate_combined_scores(
            subject_table=subject_table,
            selected_activities=selected_activities,
            weights=weights,
            activity_mean=activity_mean,
            activity_std=activity_std,
        )

        threshold, metrics = wpt.find_best_threshold(
            y_true=y_true,
            scores=combined_scores,
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

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_result = {
                "selected_activities": selected_activities,
                "activity_count": activity_count,
                "weights": weights,
                "threshold": float(threshold),
                "inner_metrics": metrics,
                "inner_scores": combined_scores,
                "inner_score_std": float(np.std(combined_scores)),
                "ranked_activities": ranked_activities,
            }

    if best_result is None:
        raise RuntimeError("Activity subset selection failed.")

    if best_result["inner_score_std"] <= EPSILON:
        best_result["inner_score_std"] = 1.0

    return best_result


# ============================================================
# Outer-test activity models
# ============================================================

def build_outer_test_matrix(
    wpt: Any,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
    selected_activities: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:

    subject_table: pd.DataFrame | None = None
    selected_features_by_activity: dict[str, list[str]] = {}

    for activity in selected_activities:
        train_indices = activity_indices(
            metadata,
            outer_train_indices,
            activity,
        )

        test_indices = activity_indices(
            metadata,
            outer_test_indices,
            activity,
        )

        model = wpt.create_model(BASE_CONFIGURATION)
        model.fit(X[train_indices], y[train_indices])

        test_scores = wpt.get_decision_scores(
            fitted_model=model,
            X=X[test_indices],
        )

        activity_subjects = aggregate_activity_scores(
            metadata=metadata.iloc[test_indices],
            y_true=y[test_indices],
            scores=test_scores,
        )

        score_column = f"score__{activity}"

        activity_subjects = activity_subjects[
            ["subject_id", "y_true", "score"]
        ].rename(columns={"score": score_column})

        if subject_table is None:
            subject_table = activity_subjects.copy()
        else:
            subject_table = subject_table.merge(
                activity_subjects[["subject_id", score_column]],
                on="subject_id",
                how="inner",
                validate="one_to_one",
            )

        selected_features_by_activity[activity] = (
            wpt.get_selected_feature_names(
                fitted_model=model,
                feature_columns=wpt_feature_columns_global,
            )
        )

        del model
        gc.collect()

    if subject_table is None:
        raise RuntimeError("No outer-test activity matrix was created.")

    return subject_table, selected_features_by_activity


# Set after loading the WPT dataset. Kept global only to avoid copying the
# 1,728 feature-name list through every function call.
wpt_feature_columns_global: list[str] = []


# ============================================================
# Main
# ============================================================

def main() -> None:
    global wpt_feature_columns_global

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wpt = load_python_module(
        WPT_SCRIPT,
        "wpt_tuning_module",
    )

    (
        metadata,
        X,
        y,
        groups,
        feature_columns,
    ) = wpt.load_wpt_dataset()

    wpt_feature_columns_global = feature_columns

    activities = sorted(
        metadata["activity"].astype(str).unique()
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
            X=np.zeros((len(y), 1), dtype=np.float32),
            y=y,
            groups=groups,
        )
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    activity_selection_rows: list[dict[str, Any]] = []
    feature_counter: Counter[tuple[str, str]] = Counter()

    print("=" * 78)
    print("NESTED WPT ACTIVITY-WEIGHTED SVM ENSEMBLE")
    print("=" * 78)
    print(f"Rows: {len(metadata)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Subjects: {metadata['subject_id'].nunique()}")
    print(f"Activities: {len(activities)}")
    print(f"Base configuration: {BASE_CONFIGURATION}")

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(outer_splits, start=1):

        print("\n" + "=" * 78)
        print(f"OUTER FOLD {outer_fold}/{N_OUTER_SPLITS}")
        print("=" * 78)

        train_subjects = set(groups[outer_train_indices])
        test_subjects = set(groups[outer_test_indices])

        if train_subjects.intersection(test_subjects):
            raise RuntimeError("Subject leakage in outer CV.")

        print(f"Train subjects: {len(train_subjects)}")
        print(f"Test subjects: {len(test_subjects)}")

        inner_cv = StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )

        inner_splits_relative = list(
            inner_cv.split(
                X=np.zeros(
                    (len(outer_train_indices), 1),
                    dtype=np.float32,
                ),
                y=y[outer_train_indices],
                groups=groups[outer_train_indices],
            )
        )

        (
            inner_subject_table,
            activity_auc,
            activity_mean,
            activity_std,
        ) = build_inner_activity_matrix(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            groups=groups,
            outer_train_indices=outer_train_indices,
            inner_splits_relative=inner_splits_relative,
            activities=activities,
        )

        selection = select_activity_subset(
            wpt=wpt,
            subject_table=inner_subject_table,
            activity_auc=activity_auc,
            activity_mean=activity_mean,
            activity_std=activity_std,
        )

        selected_activities = selection["selected_activities"]
        weights = selection["weights"]
        threshold = selection["threshold"]
        inner_metrics = selection["inner_metrics"]
        inner_score_std = selection["inner_score_std"]
        ranked_activities = selection["ranked_activities"]

        print(f"Selected activity count: {len(selected_activities)}")
        print("Selected activities: " + ", ".join(selected_activities))
        print(f"Threshold: {threshold:.6f}")
        print(f"Inner Macro-F1: {inner_metrics['macro_f1']:.4f}")
        print(
            "Inner recalls: "
            f"Healthy={inner_metrics['healthy_recall']:.4f}, "
            f"Parkinson={inner_metrics['parkinson_recall']:.4f}"
        )

        for rank, activity in enumerate(ranked_activities, start=1):
            activity_selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "activity": activity,
                    "inner_rank": rank,
                    "inner_roc_auc": activity_auc[activity],
                    "selected": activity in selected_activities,
                    "weight": weights.get(activity, 0.0),
                    "selected_activity_count": len(selected_activities),
                    "selected_threshold": threshold,
                }
            )

        (
            outer_subject_table,
            selected_features_by_activity,
        ) = build_outer_test_matrix(
            wpt=wpt,
            metadata=metadata,
            X=X,
            y=y,
            outer_train_indices=outer_train_indices,
            outer_test_indices=outer_test_indices,
            selected_activities=selected_activities,
        )

        for activity, names in selected_features_by_activity.items():
            feature_counter.update(
                (activity, feature_name)
                for feature_name in names
            )

        outer_scores = calculate_combined_scores(
            subject_table=outer_subject_table,
            selected_activities=selected_activities,
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

        outer_metrics = wpt.calculate_metrics(
            y_true=outer_y_true,
            y_pred=outer_y_pred,
            scores=outer_scores,
        )

        normalized_margin = (
            outer_scores - threshold
        ) / inner_score_std

        prediction_frame = outer_subject_table[
            ["subject_id", "y_true"]
        ].copy()

        prediction_frame["outer_fold"] = outer_fold
        prediction_frame["score"] = outer_scores
        prediction_frame["normalized_margin"] = normalized_margin
        prediction_frame["threshold"] = threshold
        prediction_frame["y_pred"] = outer_y_pred
        prediction_frame["true_label"] = (
            prediction_frame["y_true"].map(ID_TO_LABEL)
        )
        prediction_frame["predicted_label"] = (
            prediction_frame["y_pred"].map(ID_TO_LABEL)
        )
        prediction_frame["selected_activities"] = "|".join(
            selected_activities
        )

        prediction_frames.append(prediction_frame)

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "selected_activity_count": len(selected_activities),
                "selected_activities": "|".join(selected_activities),
                "selected_threshold": threshold,
                "inner_score_std": inner_score_std,
                **BASE_CONFIGURATION,
                **{
                    f"inner_{key}": value
                    for key, value in inner_metrics.items()
                },
                **{
                    f"outer_{key}": value
                    for key, value in outer_metrics.items()
                },
            }
        )

        print("\nOuter-test results:")
        print(f"Accuracy:         {outer_metrics['accuracy']:.4f}")
        print(
            "Balanced Acc:     "
            f"{outer_metrics['balanced_accuracy']:.4f}"
        )
        print(f"Macro-F1:         {outer_metrics['macro_f1']:.4f}")
        print(
            "Healthy Recall:   "
            f"{outer_metrics['healthy_recall']:.4f}"
        )
        print(
            "Parkinson Recall: "
            f"{outer_metrics['parkinson_recall']:.4f}"
        )
        print(f"ROC-AUC:          {outer_metrics['roc_auc']:.4f}")

        gc.collect()

    # ========================================================
    # Overall results
    # ========================================================

    fold_table = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    activity_selection_table = pd.DataFrame(activity_selection_rows)

    if len(predictions) != 355:
        raise RuntimeError(
            f"Expected 355 subject predictions, found {len(predictions)}."
        )

    if predictions["subject_id"].nunique() != 355:
        raise RuntimeError("Subjects are duplicated or missing.")

    y_true = predictions["y_true"].to_numpy(dtype=np.int8)
    y_pred = predictions["y_pred"].to_numpy(dtype=np.int8)
    normalized_scores = predictions[
        "normalized_margin"
    ].to_numpy(dtype=float)

    overall_metrics = wpt.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=normalized_scores,
    )

    overall_summary = pd.DataFrame(
        [
            {
                "method": "nested_wpt_activity_weighted_svm",
                "n_subjects": len(predictions),
                **overall_metrics,
                "fold_accuracy_mean": float(
                    fold_table["outer_accuracy"].mean()
                ),
                "fold_accuracy_std": float(
                    fold_table["outer_accuracy"].std()
                ),
                "fold_balanced_accuracy_mean": float(
                    fold_table["outer_balanced_accuracy"].mean()
                ),
                "fold_balanced_accuracy_std": float(
                    fold_table["outer_balanced_accuracy"].std()
                ),
                "fold_macro_f1_mean": float(
                    fold_table["outer_macro_f1"].mean()
                ),
                "fold_macro_f1_std": float(
                    fold_table["outer_macro_f1"].std()
                ),
                "fold_roc_auc_mean": float(
                    fold_table["outer_roc_auc"].mean()
                ),
                "fold_roc_auc_std": float(
                    fold_table["outer_roc_auc"].std()
                ),
            }
        ]
    )

    feature_rows = [
        {
            "activity": activity,
            "feature_name": feature_name,
            "selected_final_models": count,
        }
        for (activity, feature_name), count
        in feature_counter.most_common()
    ]

    selected_feature_table = pd.DataFrame(feature_rows)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(5.5, 4.5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Healthy", "Parkinson"],
    )

    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title("Nested WPT activity-weighted SVM")
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
        target_names=["Healthy", "Parkinson"],
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

    activity_selection_table.to_csv(
        ACTIVITY_SELECTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    selected_feature_table.to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 82)
    print("OVERALL WPT ACTIVITY-ENSEMBLE RESULTS")
    print("=" * 82)

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
                "fold_roc_auc_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nActivity selection frequency:")

    selection_frequency = (
        activity_selection_table.groupby("activity")
        .agg(
            selected_folds=("selected", "sum"),
            mean_inner_rank=("inner_rank", "mean"),
            mean_inner_auc=("inner_roc_auc", "mean"),
            mean_weight=("weight", "mean"),
        )
        .sort_values(
            ["selected_folds", "mean_inner_auc"],
            ascending=[False, False],
        )
    )

    print(
        selection_frequency.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nCreated files:")
    for output_path in [
        FOLD_RESULTS_PATH,
        ACTIVITY_SELECTION_PATH,
        SUBJECT_PREDICTIONS_PATH,
        OVERALL_SUMMARY_PATH,
        SELECTED_FEATURES_PATH,
        CONFUSION_MATRIX_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
