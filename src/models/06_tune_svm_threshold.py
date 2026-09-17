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
    balanced_accuracy_score,
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
    / "threshold_tuned_svm"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR / "fold_results.csv"
)

SUBJECT_PREDICTIONS_PATH = (
    OUTPUT_DIR / "subject_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "overall_summary.csv"
)

THRESHOLD_DETAILS_PATH = (
    OUTPUT_DIR / "threshold_details.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "subject_confusion_matrix.png"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

DEFAULT_SVM_THRESHOLD = 0.0


# ============================================================
# Import previous script
# ============================================================

def load_baseline_module():
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Baseline script not found: {BASELINE_SCRIPT}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "baseline_module",
            BASELINE_SCRIPT,
        )
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
# Metric calculation
# ============================================================

def calculate_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    return {
        "accuracy": float(
            np.mean(y_true == y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
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
# Threshold optimization
# ============================================================

def create_threshold_candidates(
    scores: np.ndarray,
) -> np.ndarray:

    unique_scores = np.unique(
        np.asarray(scores, dtype=float)
    )

    if unique_scores.size == 0:
        raise ValueError(
            "No scores available for threshold tuning."
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

    candidates = np.concatenate(
        [
            [unique_scores[0] - 1e-6],
            middle_points,
            [unique_scores[-1] + 1e-6],
            [DEFAULT_SVM_THRESHOLD],
        ]
    )

    return np.unique(candidates)


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:

    candidates = create_threshold_candidates(
        scores
    )

    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in candidates:

        y_pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = calculate_subject_metrics(
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
            -abs(float(threshold)),
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
            "Threshold optimization failed."
        )

    return best_threshold, best_metrics


# ============================================================
# Inner cross-validation
# ============================================================

def get_inner_oof_subject_predictions(
    baseline: Any,
    model_template: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_fold: int,
) -> pd.DataFrame:

    inner_cross_validator = (
        StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=(
                RANDOM_STATE + outer_fold
            ),
        )
    )

    inner_subject_frames: list[
        pd.DataFrame
    ] = []

    outer_train_indices = np.asarray(
        outer_train_indices
    )

    relative_splits = (
        inner_cross_validator.split(
            X=X[outer_train_indices],
            y=y[outer_train_indices],
            groups=groups[outer_train_indices],
        )
    )

    for inner_fold, (
        inner_train_relative,
        inner_validation_relative,
    ) in enumerate(
        relative_splits,
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
                "Subject leakage in inner CV."
            )

        model = clone(model_template)

        model.fit(
            X[inner_train_indices],
            y[inner_train_indices],
        )

        validation_scores, _ = (
            baseline.get_prediction_scores(
                model,
                X[inner_validation_indices],
            )
        )

        subject_frame = (
            baseline.aggregate_subject_predictions(
                metadata=data.iloc[
                    inner_validation_indices
                ],
                y_true=y[
                    inner_validation_indices
                ],
                scores=validation_scores,
                threshold=DEFAULT_SVM_THRESHOLD,
            )
        )

        subject_frame["inner_fold"] = (
            inner_fold
        )

        inner_subject_frames.append(
            subject_frame
        )

    inner_predictions = pd.concat(
        inner_subject_frames,
        ignore_index=True,
    )

    expected_subjects = set(
        groups[outer_train_indices]
    )

    predicted_subjects = set(
        inner_predictions["subject_id"]
    )

    if predicted_subjects != expected_subjects:
        missing = sorted(
            expected_subjects
            - predicted_subjects
        )

        duplicated = (
            inner_predictions[
                "subject_id"
            ]
            .value_counts()
        )

        duplicated = duplicated[
            duplicated != 1
        ]

        raise RuntimeError(
            "Invalid inner OOF predictions. "
            f"Missing subjects: {missing}; "
            f"duplicates: {duplicated.to_dict()}"
        )

    return inner_predictions


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

    outer_cross_validator = (
        StratifiedGroupKFold(
            n_splits=N_OUTER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[
        dict[str, Any]
    ] = []

    outer_subject_predictions: list[
        pd.DataFrame
    ] = []

    print("=" * 70)
    print("NESTED THRESHOLD-TUNED SVM")
    print("=" * 70)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Subjects: "
        f"{data['subject_id'].nunique()}"
    )

    outer_splits = (
        outer_cross_validator.split(
            X=X,
            y=y,
            groups=groups,
        )
    )

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"OUTER FOLD "
            f"{outer_fold}/{N_OUTER_SPLITS}"
        )
        print("=" * 70)

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
            f"Train subjects: "
            f"{len(train_subjects)}"
        )

        print(
            f"Test subjects: "
            f"{len(test_subjects)}"
        )

        # ----------------------------------------------------
        # Inner OOF predictions for threshold selection
        # ----------------------------------------------------

        inner_predictions = (
            get_inner_oof_subject_predictions(
                baseline=baseline,
                model_template=model_template,
                data=data,
                X=X,
                y=y,
                groups=groups,
                outer_train_indices=(
                    outer_train_indices
                ),
                outer_fold=outer_fold,
            )
        )

        inner_y_true = (
            inner_predictions[
                "y_true"
            ].to_numpy(dtype=np.int8)
        )

        inner_scores = (
            inner_predictions[
                "score"
            ].to_numpy(dtype=float)
        )

        best_threshold, inner_metrics = (
            find_best_threshold(
                y_true=inner_y_true,
                scores=inner_scores,
            )
        )

        threshold_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_threshold": (
                    best_threshold
                ),
                "inner_subjects": len(
                    inner_predictions
                ),
                **{
                    f"inner_{key}": value
                    for key, value
                    in inner_metrics.items()
                },
            }
        )

        print(
            "Selected threshold: "
            f"{best_threshold:.6f}"
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
        # Fit final model using entire outer train
        # ----------------------------------------------------

        final_model = clone(
            model_template
        )

        final_model.fit(
            X[outer_train_indices],
            y[outer_train_indices],
        )

        outer_test_scores, _ = (
            baseline.get_prediction_scores(
                final_model,
                X[outer_test_indices],
            )
        )

        # Default threshold
        default_subjects = (
            baseline.aggregate_subject_predictions(
                metadata=data.iloc[
                    outer_test_indices
                ],
                y_true=y[
                    outer_test_indices
                ],
                scores=outer_test_scores,
                threshold=(
                    DEFAULT_SVM_THRESHOLD
                ),
            )
        )

        default_metrics = (
            calculate_subject_metrics(
                y_true=default_subjects[
                    "y_true"
                ].to_numpy(),
                y_pred=default_subjects[
                    "y_pred"
                ].to_numpy(),
                scores=default_subjects[
                    "score"
                ].to_numpy(),
            )
        )

        # Tuned threshold
        tuned_subjects = (
            baseline.aggregate_subject_predictions(
                metadata=data.iloc[
                    outer_test_indices
                ],
                y_true=y[
                    outer_test_indices
                ],
                scores=outer_test_scores,
                threshold=best_threshold,
            )
        )

        tuned_metrics = (
            calculate_subject_metrics(
                y_true=tuned_subjects[
                    "y_true"
                ].to_numpy(),
                y_pred=tuned_subjects[
                    "y_pred"
                ].to_numpy(),
                scores=tuned_subjects[
                    "score"
                ].to_numpy(),
            )
        )

        tuned_subjects["outer_fold"] = (
            outer_fold
        )

        tuned_subjects[
            "selected_threshold"
        ] = best_threshold

        tuned_subjects[
            "default_prediction"
        ] = default_subjects[
            "y_pred"
        ].to_numpy()

        outer_subject_predictions.append(
            tuned_subjects
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
                "selected_threshold": (
                    best_threshold
                ),
                **{
                    f"default_{key}": value
                    for key, value
                    in default_metrics.items()
                },
                **{
                    f"tuned_{key}": value
                    for key, value
                    in tuned_metrics.items()
                },
            }
        )

        print("\nOuter-test results:")

        print(
            "Default Macro-F1: "
            f"{default_metrics['macro_f1']:.4f}"
        )

        print(
            "Tuned Macro-F1:   "
            f"{tuned_metrics['macro_f1']:.4f}"
        )

        print(
            "Tuned Accuracy:   "
            f"{tuned_metrics['accuracy']:.4f}"
        )

        print(
            "Healthy Recall:   "
            f"{tuned_metrics['healthy_recall']:.4f}"
        )

        print(
            "Parkinson Recall: "
            f"{tuned_metrics['parkinson_recall']:.4f}"
        )

    # ========================================================
    # Combine all outer-test predictions
    # ========================================================

    predictions = pd.concat(
        outer_subject_predictions,
        ignore_index=True,
    )

    if predictions["subject_id"].nunique() != 355:
        raise RuntimeError(
            "Not all subjects received an "
            "outer-fold prediction."
        )

    if len(predictions) != 355:
        raise RuntimeError(
            "Subjects were duplicated in "
            "outer predictions."
        )

    y_true = predictions[
        "y_true"
    ].to_numpy(dtype=np.int8)

    tuned_y_pred = predictions[
        "y_pred"
    ].to_numpy(dtype=np.int8)

    default_y_pred = predictions[
        "default_prediction"
    ].to_numpy(dtype=np.int8)

    scores = predictions[
        "score"
    ].to_numpy(dtype=float)

    tuned_overall_metrics = (
        calculate_subject_metrics(
            y_true=y_true,
            y_pred=tuned_y_pred,
            scores=scores,
        )
    )

    default_overall_metrics = (
        calculate_subject_metrics(
            y_true=y_true,
            y_pred=default_y_pred,
            scores=scores,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "method": "default_threshold",
                **default_overall_metrics,
            },
            {
                "method": "nested_tuned_threshold",
                **tuned_overall_metrics,
            },
        ]
    )

    fold_table = pd.DataFrame(
        fold_rows
    )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    # ========================================================
    # Save confusion matrix
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        tuned_y_pred,
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
        "Threshold-tuned SVM — Subject level"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ========================================================
    # Save tables
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

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_table.to_csv(
        THRESHOLD_DETAILS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("OVERALL OUTER-FOLD RESULTS")
    print("=" * 70)

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

    print("\nCreated files:")

    for path in [
        FOLD_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        SUMMARY_PATH,
        THRESHOLD_DETAILS_PATH,
        CONFUSION_MATRIX_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()