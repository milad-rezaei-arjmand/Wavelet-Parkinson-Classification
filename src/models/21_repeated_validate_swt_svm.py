from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
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
from sklearn.svm import SVC
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SWT_FEATURE_PATH = (
    OUTPUTS_DIR / "swt_features_db4_level5.parquet"
)

RESULTS_DIR = (
    OUTPUTS_DIR / "repeated_swt_validation"
)

FOLD_RESULTS_PATH = (
    RESULTS_DIR / "fold_results.csv"
)

REPEAT_RESULTS_PATH = (
    RESULTS_DIR / "repeat_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "repeated_subject_predictions.csv"
)

SUBJECT_STABILITY_PATH = (
    RESULTS_DIR / "subject_stability.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIR / "overall_summary.csv"
)

CONSENSUS_REPORT_PATH = (
    RESULTS_DIR / "consensus_classification_report.txt"
)

CONSENSUS_MATRIX_PATH = (
    RESULTS_DIR / "consensus_confusion_matrix.png"
)


# ============================================================
# Configuration
# ============================================================

BASE_RANDOM_STATE = 42

N_REPEATS = 5
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

SELECT_K = 300
SVM_C = 10.0
SVM_GAMMA = "scale"

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

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "healthy_precision",
    "healthy_recall",
    "healthy_f1",
    "parkinson_precision",
    "parkinson_recall",
    "parkinson_f1",
    "roc_auc",
]


# ============================================================
# Data
# ============================================================

def load_dataset() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:

    if not SWT_FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"SWT feature file not found: {SWT_FEATURE_PATH}"
        )

    print("Loading SWT features...")

    data = pd.read_parquet(SWT_FEATURE_PATH)

    missing_metadata = set(META_COLUMNS) - set(data.columns)

    if missing_metadata:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_metadata)}"
        )

    data["subject_id"] = (
        data["subject_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(3)
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
            f"Expected 15975 rows, found {len(data)}."
        )

    if data["subject_id"].nunique() != 355:
        raise ValueError("Expected exactly 355 subjects.")

    if data["activity"].nunique() != 11:
        raise ValueError("Expected exactly 11 activities.")

    windows_per_subject = data.groupby("subject_id").size()

    if not np.all(windows_per_subject == 45):
        raise ValueError(
            "Not every subject has exactly 45 windows."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in META_COLUMNS
    ]

    if len(feature_columns) != 648:
        raise ValueError(
            f"Expected 648 SWT features, found {len(feature_columns)}."
        )

    print("Converting SWT features to NumPy...")

    X = data[feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN or Inf.")

    unknown_labels = (
        set(data["label"].unique())
        - set(LABEL_TO_ID)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {sorted(unknown_labels)}"
        )

    y = (
        data["label"]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    groups = data["subject_id"].to_numpy()

    metadata = data[META_COLUMNS].copy()

    del data
    gc.collect()

    return metadata, X, y, groups, feature_columns


# ============================================================
# Model
# ============================================================

def create_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "variance",
                VarianceThreshold(threshold=0.0),
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
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=SVM_C,
                    gamma=SVM_GAMMA,
                    class_weight="balanced",
                    cache_size=1024,
                ),
            ),
        ]
    )


def decision_scores(
    fitted_model: Pipeline,
    X: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(
        fitted_model.decision_function(X),
        dtype=float,
    ).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Decision scores contain NaN or Inf."
        )

    return scores


# ============================================================
# Subject aggregation
# ============================================================

def aggregate_subject_scores(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:

    temporary = metadata[
        ["subject_id", "activity"]
    ].copy()

    temporary["y_true"] = np.asarray(
        y_true,
        dtype=np.int8,
    )

    temporary["score"] = np.asarray(
        scores,
        dtype=float,
    )

    activity_label_counts = (
        temporary.groupby(
            ["subject_id", "activity"]
        )["y_true"]
        .nunique()
    )

    if not np.all(activity_label_counts == 1):
        raise ValueError(
            "Inconsistent labels inside a subject/activity group."
        )

    activity_scores = (
        temporary.groupby(
            ["subject_id", "activity"],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            activity_score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    subject_scores = (
        activity_scores.groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            score=("activity_score", "mean"),
            n_activities=("activity", "nunique"),
            n_windows=("n_windows", "sum"),
        )
    )

    if not np.all(subject_scores["n_activities"] == 11):
        raise ValueError(
            "Not every subject has 11 activities."
        )

    if not np.all(subject_scores["n_windows"] == 45):
        raise ValueError(
            "Not every subject has 45 windows."
        )

    return subject_scores


# ============================================================
# Metrics and threshold
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
            balanced_accuracy_score(y_true, y_pred)
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
            roc_auc_score(y_true, scores)
        ),
    }


def threshold_candidates(
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
        return np.asarray(
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
                [unique_scores[0] - 1e-6],
                middle_points,
                [unique_scores[-1] + 1e-6],
                [0.0],
            ]
        )
    )


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:

    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in threshold_candidates(scores):

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
            metrics["healthy_recall"],
            metrics["roc_auc"],
            -abs(float(threshold)),
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError(
            "Threshold selection failed."
        )

    return best_threshold, best_metrics


# ============================================================
# Confidence intervals
# ============================================================

def mean_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:

    values = np.asarray(values, dtype=float)

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1))

    if values.size < 2:
        return (
            mean_value,
            std_value,
            mean_value,
            mean_value,
        )

    standard_error = (
        std_value / np.sqrt(values.size)
    )

    critical_value = float(
        t.ppf(
            (1.0 + confidence) / 2.0,
            df=values.size - 1,
        )
    )

    half_width = (
        critical_value * standard_error
    )

    return (
        mean_value,
        std_value,
        mean_value - half_width,
        mean_value + half_width,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata, X, y, groups, _ = load_dataset()

    model_template = create_model()

    fold_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    print("\n" + "=" * 88)
    print("REPEATED NESTED SUBJECT-GROUPED VALIDATION — SWT SVM")
    print("=" * 88)

    print(f"\nRows: {len(metadata)}")
    print(
        f"Subjects: {metadata['subject_id'].nunique()}"
    )
    print(f"SWT features: {X.shape[1]}")
    print(f"Selected features: {SELECT_K}")
    print(f"Repeats: {N_REPEATS}")
    print(f"Outer folds per repeat: {N_OUTER_SPLITS}")
    print(f"Inner folds: {N_INNER_SPLITS}")
    print(f"Total outer evaluations: {N_REPEATS * N_OUTER_SPLITS}")

    repeat_progress = tqdm(
        range(1, N_REPEATS + 1),
        desc="Repeated validation",
        unit="repeat",
    )

    for repeat_number in repeat_progress:

        outer_random_state = (
            BASE_RANDOM_STATE
            + (repeat_number - 1) * 1000
        )

        outer_cv = StratifiedGroupKFold(
            n_splits=N_OUTER_SPLITS,
            shuffle=True,
            random_state=outer_random_state,
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

        current_repeat_predictions: list[
            pd.DataFrame
        ] = []

        for outer_fold, (
            outer_train_indices,
            outer_test_indices,
        ) in enumerate(
            outer_splits,
            start=1,
        ):

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

            inner_random_state = (
                outer_random_state
                + outer_fold
            )

            inner_cv = StratifiedGroupKFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=inner_random_state,
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
                    groups=groups[outer_train_indices],
                )
            )

            inner_subject_frames: list[
                pd.DataFrame
            ] = []

            for (
                inner_train_relative,
                inner_validation_relative,
            ) in inner_splits_relative:

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

                inner_train_subjects = set(
                    groups[inner_train_indices]
                )

                inner_validation_subjects = set(
                    groups[inner_validation_indices]
                )

                if inner_train_subjects.intersection(
                    inner_validation_subjects
                ):
                    raise RuntimeError(
                        "Subject leakage in inner CV."
                    )

                model = clone(model_template)

                model.fit(
                    X[inner_train_indices],
                    y[inner_train_indices],
                )

                validation_window_scores = (
                    decision_scores(
                        fitted_model=model,
                        X=X[inner_validation_indices],
                    )
                )

                validation_subject_scores = (
                    aggregate_subject_scores(
                        metadata=metadata.iloc[
                            inner_validation_indices
                        ],
                        y_true=y[
                            inner_validation_indices
                        ],
                        scores=validation_window_scores,
                    )
                )

                inner_subject_frames.append(
                    validation_subject_scores
                )

                del model
                gc.collect()

            inner_predictions = pd.concat(
                inner_subject_frames,
                ignore_index=True,
            )

            expected_inner_subjects = set(
                groups[outer_train_indices]
            )

            actual_inner_subjects = set(
                inner_predictions["subject_id"]
            )

            if actual_inner_subjects != expected_inner_subjects:
                raise RuntimeError(
                    "Inner OOF subject set is incomplete."
                )

            if (
                inner_predictions["subject_id"]
                .value_counts()
                .ne(1)
                .any()
            ):
                raise RuntimeError(
                    "Duplicated subjects in inner OOF."
                )

            inner_y_true = (
                inner_predictions["y_true"]
                .to_numpy(dtype=np.int8)
            )

            inner_scores = (
                inner_predictions["score"]
                .to_numpy(dtype=float)
            )

            selected_threshold, inner_metrics = (
                select_threshold(
                    y_true=inner_y_true,
                    scores=inner_scores,
                )
            )

            inner_score_std = float(
                np.std(inner_scores)
            )

            if inner_score_std <= EPSILON:
                inner_score_std = 1.0

            final_model = clone(model_template)

            final_model.fit(
                X[outer_train_indices],
                y[outer_train_indices],
            )

            outer_window_scores = decision_scores(
                fitted_model=final_model,
                X=X[outer_test_indices],
            )

            outer_subjects = aggregate_subject_scores(
                metadata=metadata.iloc[
                    outer_test_indices
                ],
                y_true=y[outer_test_indices],
                scores=outer_window_scores,
            )

            outer_subjects["y_pred"] = (
                outer_subjects["score"]
                >= selected_threshold
            ).astype(np.int8)

            outer_subjects["normalized_margin"] = (
                (
                    outer_subjects["score"]
                    - selected_threshold
                )
                / inner_score_std
            )

            outer_metrics = calculate_metrics(
                y_true=outer_subjects[
                    "y_true"
                ].to_numpy(dtype=np.int8),
                y_pred=outer_subjects[
                    "y_pred"
                ].to_numpy(dtype=np.int8),
                scores=outer_subjects[
                    "score"
                ].to_numpy(dtype=float),
            )

            fold_rows.append(
                {
                    "repeat": repeat_number,
                    "outer_fold": outer_fold,
                    "outer_random_state": (
                        outer_random_state
                    ),
                    "inner_random_state": (
                        inner_random_state
                    ),
                    "train_subjects": len(
                        train_subjects
                    ),
                    "test_subjects": len(
                        test_subjects
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

            outer_subjects["repeat"] = (
                repeat_number
            )

            outer_subjects["outer_fold"] = (
                outer_fold
            )

            outer_subjects[
                "selected_threshold"
            ] = selected_threshold

            outer_subjects["true_label"] = (
                outer_subjects["y_true"]
                .map(ID_TO_LABEL)
            )

            outer_subjects["predicted_label"] = (
                outer_subjects["y_pred"]
                .map(ID_TO_LABEL)
            )

            current_repeat_predictions.append(
                outer_subjects
            )

            del final_model
            del inner_predictions
            del inner_subject_frames

            gc.collect()

        repeat_predictions = pd.concat(
            current_repeat_predictions,
            ignore_index=True,
        )

        if len(repeat_predictions) != 355:
            raise RuntimeError(
                "Each repeat must predict exactly 355 subjects."
            )

        if (
            repeat_predictions["subject_id"]
            .nunique()
            != 355
        ):
            raise RuntimeError(
                "Subjects duplicated or missing in a repeat."
            )

        repeat_metrics = calculate_metrics(
            y_true=repeat_predictions[
                "y_true"
            ].to_numpy(dtype=np.int8),
            y_pred=repeat_predictions[
                "y_pred"
            ].to_numpy(dtype=np.int8),
            scores=repeat_predictions[
                "normalized_margin"
            ].to_numpy(dtype=float),
        )

        repeat_matrix = confusion_matrix(
            repeat_predictions["y_true"],
            repeat_predictions["y_pred"],
            labels=[0, 1],
        )

        repeat_rows.append(
            {
                "repeat": repeat_number,
                "outer_random_state": (
                    outer_random_state
                ),
                **repeat_metrics,
                "healthy_correct": int(
                    repeat_matrix[0, 0]
                ),
                "healthy_incorrect": int(
                    repeat_matrix[0, 1]
                ),
                "parkinson_incorrect": int(
                    repeat_matrix[1, 0]
                ),
                "parkinson_correct": int(
                    repeat_matrix[1, 1]
                ),
            }
        )

        prediction_frames.append(
            repeat_predictions
        )

        repeat_progress.set_postfix(
            accuracy=(
                f"{repeat_metrics['accuracy']:.4f}"
            ),
            macro_f1=(
                f"{repeat_metrics['macro_f1']:.4f}"
            ),
        )

    fold_table = pd.DataFrame(fold_rows)
    repeat_table = pd.DataFrame(repeat_rows)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    expected_prediction_rows = (
        355 * N_REPEATS
    )

    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            "Unexpected repeated prediction count: "
            f"{len(predictions)}"
        )

    # ========================================================
    # Repeated-performance summary
    # ========================================================

    summary_row: dict[str, Any] = {
        "method": (
            "repeated_nested_swt_svm"
        ),
        "n_repeats": N_REPEATS,
        "n_subjects": 355,
        "outer_folds_per_repeat": (
            N_OUTER_SPLITS
        ),
        "inner_folds": N_INNER_SPLITS,
        "selected_features": SELECT_K,
        "C": SVM_C,
        "gamma": SVM_GAMMA,
    }

    for metric in METRIC_COLUMNS:

        (
            mean_value,
            std_value,
            ci_low,
            ci_high,
        ) = mean_confidence_interval(
            repeat_table[
                metric
            ].to_numpy(dtype=float)
        )

        summary_row[
            f"{metric}_mean"
        ] = mean_value

        summary_row[
            f"{metric}_std"
        ] = std_value

        summary_row[
            f"{metric}_ci95_low"
        ] = ci_low

        summary_row[
            f"{metric}_ci95_high"
        ] = ci_high

    summary_table = pd.DataFrame(
        [summary_row]
    )

    # ========================================================
    # Subject stability and consensus
    # ========================================================

    predictions["is_correct"] = (
        predictions["y_true"]
        == predictions["y_pred"]
    ).astype(np.int8)

    subject_stability = (
        predictions.groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            true_label=(
                "true_label",
                "first",
            ),
            n_predictions=(
                "y_pred",
                "size",
            ),
            correct_count=(
                "is_correct",
                "sum",
            ),
            parkinson_prediction_rate=(
                "y_pred",
                "mean",
            ),
            mean_normalized_margin=(
                "normalized_margin",
                "mean",
            ),
            std_normalized_margin=(
                "normalized_margin",
                "std",
            ),
            minimum_normalized_margin=(
                "normalized_margin",
                "min",
            ),
            maximum_normalized_margin=(
                "normalized_margin",
                "max",
            ),
        )
    )

    subject_stability[
        "incorrect_count"
    ] = (
        subject_stability[
            "n_predictions"
        ]
        - subject_stability[
            "correct_count"
        ]
    )

    subject_stability[
        "error_rate"
    ] = (
        subject_stability[
            "incorrect_count"
        ]
        / subject_stability[
            "n_predictions"
        ]
    )

    subject_stability[
        "consensus_y_pred"
    ] = (
        subject_stability[
            "mean_normalized_margin"
        ] >= 0.0
    ).astype(np.int8)

    subject_stability[
        "consensus_predicted_label"
    ] = (
        subject_stability[
            "consensus_y_pred"
        ].map(ID_TO_LABEL)
    )

    subject_stability[
        "consensus_correct"
    ] = (
        subject_stability[
            "consensus_y_pred"
        ]
        == subject_stability[
            "y_true"
        ]
    )

    subject_stability = (
        subject_stability.sort_values(
            by=[
                "error_rate",
                "std_normalized_margin",
                "subject_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    consensus_y_true = (
        subject_stability["y_true"]
        .to_numpy(dtype=np.int8)
    )

    consensus_y_pred = (
        subject_stability[
            "consensus_y_pred"
        ].to_numpy(dtype=np.int8)
    )

    consensus_scores = (
        subject_stability[
            "mean_normalized_margin"
        ].to_numpy(dtype=float)
    )

    consensus_metrics = calculate_metrics(
        y_true=consensus_y_true,
        y_pred=consensus_y_pred,
        scores=consensus_scores,
    )

    consensus_matrix = confusion_matrix(
        consensus_y_true,
        consensus_y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        consensus_y_true,
        consensus_y_pred,
        labels=[0, 1],
        target_names=[
            "Healthy",
            "Parkinson",
        ],
        digits=4,
        zero_division=0,
    )

    CONSENSUS_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    figure, axis = plt.subplots(
        figsize=(5.5, 4.5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=consensus_matrix,
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
        "Repeated-CV consensus SWT-SVM"
    )

    figure.tight_layout()

    figure.savefig(
        CONSENSUS_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ========================================================
    # Save
    # ========================================================

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    repeat_table.to_csv(
        REPEAT_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    subject_stability.to_csv(
        SUBJECT_STABILITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_table.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console
    # ========================================================

    print("\n" + "=" * 102)
    print("REPEATED VALIDATION RESULTS")
    print("=" * 102)

    print(
        repeat_table[
            [
                "repeat",
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

    print("\nMean ± standard deviation and 95% CI:")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "healthy_recall",
        "parkinson_recall",
        "roc_auc",
    ]:

        print(
            f"{metric:22s} "
            f"{summary_row[f'{metric}_mean']:.4f} "
            f"± {summary_row[f'{metric}_std']:.4f} "
            f"[{summary_row[f'{metric}_ci95_low']:.4f}, "
            f"{summary_row[f'{metric}_ci95_high']:.4f}]"
        )

    print("\nConsensus across repeats:")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "healthy_recall",
        "parkinson_recall",
        "roc_auc",
    ]:
        print(
            f"{metric:22s} "
            f"{consensus_metrics[metric]:.4f}"
        )

    print("\nConsensus confusion matrix:")
    print(consensus_matrix)

    print("\nMost unstable / difficult subjects:")

    print(
        subject_stability[
            [
                "subject_id",
                "true_label",
                "incorrect_count",
                "error_rate",
                "parkinson_prediction_rate",
                "mean_normalized_margin",
                "std_normalized_margin",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nCreated files:")

    for output_path in [
        FOLD_RESULTS_PATH,
        REPEAT_RESULTS_PATH,
        PREDICTIONS_PATH,
        SUBJECT_STABILITY_PATH,
        OVERALL_SUMMARY_PATH,
        CONSENSUS_REPORT_PATH,
        CONSENSUS_MATRIX_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
