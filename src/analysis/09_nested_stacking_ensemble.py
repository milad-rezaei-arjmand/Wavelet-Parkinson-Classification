from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_SCRIPT = (
    PROJECT_ROOT / "05_train_baselines.py"
)

ENSEMBLE_SCRIPT = (
    PROJECT_ROOT / "08_nested_activity_ensemble.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "nested_stacking_ensemble"
)

FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"

SUBJECT_PREDICTIONS_PATH = (
    OUTPUT_DIR / "subject_predictions.csv"
)

SEARCH_RESULTS_PATH = (
    OUTPUT_DIR / "meta_search_results.csv"
)

COEFFICIENTS_PATH = (
    OUTPUT_DIR / "meta_coefficients.csv"
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
N_META_SPLITS = 4

TOP_ACTIVITY_COUNTS = [
    3,
    5,
    7,
    9,
    11,
]

META_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

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
            f"Python file not found: {file_path}"
        )

    specification = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not load module: {file_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


# ============================================================
# Meta-classifier
# ============================================================

def create_meta_model(
    regularization_c: float,
    random_state: int,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=regularization_c,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_meta_oof_scores(
    X_meta: np.ndarray,
    y_meta: np.ndarray,
    regularization_c: float,
    random_state: int,
) -> np.ndarray:

    cross_validator = StratifiedKFold(
        n_splits=N_META_SPLITS,
        shuffle=True,
        random_state=random_state,
    )

    oof_scores = np.full(
        len(y_meta),
        fill_value=np.nan,
        dtype=float,
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(
            X_meta,
            y_meta,
        ),
        start=1,
    ):
        model = create_meta_model(
            regularization_c=regularization_c,
            random_state=(
                random_state + fold_number
            ),
        )

        model.fit(
            X_meta[train_indices],
            y_meta[train_indices],
        )

        validation_scores = model.decision_function(
            X_meta[validation_indices]
        )

        oof_scores[validation_indices] = (
            validation_scores
        )

    if not np.all(np.isfinite(oof_scores)):
        raise RuntimeError(
            "Meta OOF scores contain invalid values."
        )

    return oof_scores


# ============================================================
# Meta-model selection
# ============================================================

def select_meta_configuration(
    ensemble: Any,
    subject_table: pd.DataFrame,
    ranked_activities: list[str],
    outer_fold: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    y_true = subject_table[
        "y_true"
    ].to_numpy(dtype=np.int8)

    search_rows: list[dict[str, Any]] = []

    best_configuration: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    for activity_count in TOP_ACTIVITY_COUNTS:

        if activity_count > len(ranked_activities):
            continue

        selected_activities = ranked_activities[
            :activity_count
        ]

        score_columns = [
            f"score__{activity}"
            for activity in selected_activities
        ]

        X_meta = subject_table[
            score_columns
        ].to_numpy(dtype=float)

        for regularization_c in META_C_VALUES:

            meta_oof_scores = get_meta_oof_scores(
                X_meta=X_meta,
                y_meta=y_true,
                regularization_c=regularization_c,
                random_state=(
                    RANDOM_STATE
                    + outer_fold * 100
                    + activity_count
                ),
            )

            threshold, metrics = (
                ensemble.find_best_threshold(
                    y_true=y_true,
                    scores=meta_oof_scores,
                )
            )

            minimum_class_recall = min(
                metrics["healthy_recall"],
                metrics["parkinson_recall"],
            )

            search_rows.append(
                {
                    "outer_fold": outer_fold,
                    "activity_count": activity_count,
                    "selected_activities": "|".join(
                        selected_activities
                    ),
                    "meta_c": regularization_c,
                    "threshold": threshold,
                    **metrics,
                }
            )

            selection_key = (
                metrics["macro_f1"],
                minimum_class_recall,
                metrics["balanced_accuracy"],
                metrics["roc_auc"],
                -activity_count,
                -abs(np.log10(regularization_c)),
            )

            if (
                best_key is None
                or selection_key > best_key
            ):
                best_key = selection_key

                best_configuration = {
                    "activity_count": activity_count,
                    "selected_activities": (
                        selected_activities
                    ),
                    "score_columns": score_columns,
                    "meta_c": regularization_c,
                    "threshold": threshold,
                    "inner_metrics": metrics,
                    "inner_oof_scores": (
                        meta_oof_scores
                    ),
                }

    if best_configuration is None:
        raise RuntimeError(
            "Meta-model configuration selection failed."
        )

    return best_configuration, search_rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_python_module(
        BASELINE_SCRIPT,
        "baseline_module",
    )

    ensemble = load_python_module(
        ENSEMBLE_SCRIPT,
        "ensemble_module",
    )

    (
        data,
        X,
        y,
        groups,
        feature_columns,
    ) = baseline.load_dataset()

    base_model_template = (
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

    outer_cross_validator = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    search_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    print("=" * 75)
    print("NESTED STACKED ACTIVITY-SVM ENSEMBLE")
    print("=" * 75)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Subjects: "
        f"{data['subject_id'].nunique()}"
    )
    print(f"Activities: {len(activities)}")

    outer_splits = outer_cross_validator.split(
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
                "Subject leakage detected."
            )

        print(
            f"Train subjects: "
            f"{len(train_subjects)}"
        )

        print(
            f"Test subjects: "
            f"{len(test_subjects)}"
        )

        # ----------------------------------------------------
        # Inner out-of-fold activity scores
        # ----------------------------------------------------

        (
            inner_subject_table,
            activity_auc,
            _,
            _,
        ) = ensemble.build_inner_activity_matrix(
            baseline=baseline,
            model_template=base_model_template,
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

        ranked_activities = sorted(
            activities,
            key=activity_auc.get,
            reverse=True,
        )

        (
            selected_configuration,
            current_search_rows,
        ) = select_meta_configuration(
            ensemble=ensemble,
            subject_table=inner_subject_table,
            ranked_activities=ranked_activities,
            outer_fold=outer_fold,
        )

        search_rows.extend(
            current_search_rows
        )

        selected_activities = (
            selected_configuration[
                "selected_activities"
            ]
        )

        score_columns = (
            selected_configuration[
                "score_columns"
            ]
        )

        meta_c = selected_configuration[
            "meta_c"
        ]

        threshold = (
            selected_configuration[
                "threshold"
            ]
        )

        inner_metrics = (
            selected_configuration[
                "inner_metrics"
            ]
        )

        print(
            "Selected activity count: "
            f"{len(selected_activities)}"
        )

        print(
            "Selected activities: "
            + ", ".join(selected_activities)
        )

        print(
            f"Selected meta C: {meta_c}"
        )

        print(
            "Selected threshold: "
            f"{threshold:.6f}"
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

        # ----------------------------------------------------
        # Fit final meta-model
        # ----------------------------------------------------

        inner_X_meta = inner_subject_table[
            score_columns
        ].to_numpy(dtype=float)

        inner_y_meta = inner_subject_table[
            "y_true"
        ].to_numpy(dtype=np.int8)

        final_meta_model = create_meta_model(
            regularization_c=meta_c,
            random_state=(
                RANDOM_STATE + outer_fold
            ),
        )

        final_meta_model.fit(
            inner_X_meta,
            inner_y_meta,
        )

        # ----------------------------------------------------
        # Base-model scores for outer test
        # ----------------------------------------------------

        outer_subject_table = (
            ensemble.build_outer_test_matrix(
                baseline=baseline,
                model_template=(
                    base_model_template
                ),
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

        outer_X_meta = outer_subject_table[
            score_columns
        ].to_numpy(dtype=float)

        outer_scores = (
            final_meta_model.decision_function(
                outer_X_meta
            )
        )

        outer_y_true = outer_subject_table[
            "y_true"
        ].to_numpy(dtype=np.int8)

        outer_y_pred = (
            outer_scores >= threshold
        ).astype(np.int8)

        outer_metrics = (
            ensemble.calculate_metrics(
                y_true=outer_y_true,
                y_pred=outer_y_pred,
                scores=outer_scores,
            )
        )

        # ----------------------------------------------------
        # Meta-model coefficients
        # ----------------------------------------------------

        logistic_regression = (
            final_meta_model.named_steps[
                "classifier"
            ]
        )

        coefficients = (
            logistic_regression.coef_[0]
        )

        for activity, coefficient in zip(
            selected_activities,
            coefficients,
        ):
            coefficient_rows.append(
                {
                    "outer_fold": outer_fold,
                    "activity": activity,
                    "inner_activity_auc": (
                        activity_auc[activity]
                    ),
                    "coefficient": float(
                        coefficient
                    ),
                    "absolute_coefficient": float(
                        abs(coefficient)
                    ),
                    "selected_activity_count": (
                        len(selected_activities)
                    ),
                    "meta_c": meta_c,
                    "threshold": threshold,
                }
            )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

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

        prediction_frame["score"] = (
            outer_scores
        )

        prediction_frame["threshold"] = (
            threshold
        )

        prediction_frame["y_pred"] = (
            outer_y_pred
        )

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

        prediction_frame["meta_c"] = meta_c

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
                "meta_c": meta_c,
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
    # Overall results
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

    overall_metrics = (
        ensemble.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
        )
    )

    overall_summary = pd.DataFrame(
        [
            {
                "method": (
                    "nested_stacked_activity_svm"
                ),
                "n_subjects": len(predictions),
                **overall_metrics,
            }
        ]
    )

    fold_table = pd.DataFrame(
        fold_rows
    )

    search_table = pd.DataFrame(
        search_rows
    )

    coefficients_table = pd.DataFrame(
        coefficient_rows
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
        "Nested stacked activity-SVM ensemble"
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

    search_table.to_csv(
        SEARCH_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients_table.to_csv(
        COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console output
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

    print("\nMean absolute meta coefficients:")

    coefficient_summary = (
        coefficients_table.groupby(
            "activity"
        )
        .agg(
            selected_folds=(
                "outer_fold",
                "nunique",
            ),
            mean_coefficient=(
                "coefficient",
                "mean",
            ),
            mean_absolute_coefficient=(
                "absolute_coefficient",
                "mean",
            ),
            mean_inner_auc=(
                "inner_activity_auc",
                "mean",
            ),
        )
        .sort_values(
            "mean_absolute_coefficient",
            ascending=False,
        )
    )

    print(
        coefficient_summary.to_string(
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    print("\nCreated files:")

    for output_path in [
        FOLD_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        SEARCH_RESULTS_PATH,
        COEFFICIENTS_PATH,
        OVERALL_SUMMARY_PATH,
        CONFUSION_MATRIX_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()