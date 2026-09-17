from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_results"
)

THRESHOLD_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "threshold_tuned_svm"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "activity_analysis"
)

WINDOW_PREDICTIONS_PATH = (
    BASELINE_DIR
    / "baseline_window_predictions.parquet"
)

THRESHOLD_DETAILS_PATH = (
    THRESHOLD_DIR
    / "threshold_details.csv"
)

ACTIVITY_SCORES_PATH = (
    OUTPUT_DIR
    / "subject_activity_scores.csv"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIR
    / "activity_fold_metrics.csv"
)

ACTIVITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "activity_summary.csv"
)


# ============================================================
# Settings
# ============================================================

MODEL_NAME = "SVM_RBF"

ID_TO_LABEL = {
    0: "Healthy",
    1: "Parkinson",
}


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_healthy = int(matrix[0, 0])
    false_parkinson = int(matrix[0, 1])
    false_healthy = int(matrix[1, 0])
    true_parkinson = int(matrix[1, 1])

    if len(np.unique(y_true)) == 2:
        auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )
    else:
        auc = np.nan

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
        "roc_auc": auc,
        "true_healthy": true_healthy,
        "false_parkinson": false_parkinson,
        "false_healthy": false_healthy,
        "true_parkinson": true_parkinson,
    }


# ============================================================
# Load and aggregate predictions
# ============================================================

def load_subject_activity_scores() -> pd.DataFrame:

    if not WINDOW_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{WINDOW_PREDICTIONS_PATH}"
        )

    if not THRESHOLD_DETAILS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{THRESHOLD_DETAILS_PATH}"
        )

    predictions = pd.read_parquet(
        WINDOW_PREDICTIONS_PATH
    )

    predictions = predictions[
        predictions["model"] == MODEL_NAME
    ].copy()

    if predictions.empty:
        raise RuntimeError(
            f"No predictions found for {MODEL_NAME}."
        )

    predictions["subject_id"] = (
        predictions["subject_id"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(3)
    )

    required_columns = {
        "subject_id",
        "activity",
        "fold",
        "y_true",
        "score",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    subject_activity_scores = (
        predictions.groupby(
            [
                "subject_id",
                "activity",
                "fold",
            ],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    thresholds = pd.read_csv(
        THRESHOLD_DETAILS_PATH
    )

    thresholds = thresholds[
        [
            "outer_fold",
            "selected_threshold",
        ]
    ].rename(
        columns={
            "outer_fold": "fold",
        }
    )

    subject_activity_scores = (
        subject_activity_scores.merge(
            thresholds,
            on="fold",
            how="left",
            validate="many_to_one",
        )
    )

    if subject_activity_scores[
        "selected_threshold"
    ].isna().any():
        raise ValueError(
            "Some folds have no selected threshold."
        )

    subject_activity_scores["y_pred"] = (
        subject_activity_scores["score"]
        >= subject_activity_scores[
            "selected_threshold"
        ]
    ).astype(np.int8)

    subject_activity_scores["true_label"] = (
        subject_activity_scores[
            "y_true"
        ].map(ID_TO_LABEL)
    )

    subject_activity_scores[
        "predicted_label"
    ] = (
        subject_activity_scores[
            "y_pred"
        ].map(ID_TO_LABEL)
    )

    return subject_activity_scores


# ============================================================
# Fold-level evaluation
# ============================================================

def calculate_fold_metrics(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    grouped = scores.groupby(
        [
            "activity",
            "fold",
        ],
        sort=True,
    )

    for (
        activity,
        fold,
    ), fold_data in grouped:

        y_true = fold_data[
            "y_true"
        ].to_numpy(dtype=np.int8)

        y_pred = fold_data[
            "y_pred"
        ].to_numpy(dtype=np.int8)

        activity_scores = fold_data[
            "score"
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=activity_scores,
        )

        healthy_scores = fold_data.loc[
            fold_data["y_true"] == 0,
            "score",
        ]

        parkinson_scores = fold_data.loc[
            fold_data["y_true"] == 1,
            "score",
        ]

        rows.append(
            {
                "activity": activity,
                "fold": int(fold),
                "n_subjects": len(fold_data),
                "n_healthy": int(
                    np.sum(y_true == 0)
                ),
                "n_parkinson": int(
                    np.sum(y_true == 1)
                ),
                "healthy_score_mean": float(
                    healthy_scores.mean()
                ),
                "parkinson_score_mean": float(
                    parkinson_scores.mean()
                ),
                "score_mean_difference": float(
                    parkinson_scores.mean()
                    - healthy_scores.mean()
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Overall activity summary
# ============================================================

def calculate_activity_summary(
    scores: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for activity, activity_data in scores.groupby(
        "activity",
        sort=True,
    ):

        y_true = activity_data[
            "y_true"
        ].to_numpy(dtype=np.int8)

        y_pred = activity_data[
            "y_pred"
        ].to_numpy(dtype=np.int8)

        activity_scores = activity_data[
            "score"
        ].to_numpy(dtype=float)

        pooled_metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=activity_scores,
        )

        current_folds = fold_metrics[
            fold_metrics["activity"]
            == activity
        ]

        healthy_scores = activity_data.loc[
            activity_data["y_true"] == 0,
            "score",
        ]

        parkinson_scores = activity_data.loc[
            activity_data["y_true"] == 1,
            "score",
        ]

        rows.append(
            {
                "activity": activity,
                "n_subjects": len(
                    activity_data
                ),
                "n_healthy": int(
                    np.sum(y_true == 0)
                ),
                "n_parkinson": int(
                    np.sum(y_true == 1)
                ),
                "healthy_score_mean": float(
                    healthy_scores.mean()
                ),
                "healthy_score_std": float(
                    healthy_scores.std()
                ),
                "parkinson_score_mean": float(
                    parkinson_scores.mean()
                ),
                "parkinson_score_std": float(
                    parkinson_scores.std()
                ),
                "score_mean_difference": float(
                    parkinson_scores.mean()
                    - healthy_scores.mean()
                ),
                "pooled_accuracy": (
                    pooled_metrics["accuracy"]
                ),
                "pooled_balanced_accuracy": (
                    pooled_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "pooled_macro_f1": (
                    pooled_metrics["macro_f1"]
                ),
                "pooled_healthy_recall": (
                    pooled_metrics[
                        "healthy_recall"
                    ]
                ),
                "pooled_parkinson_recall": (
                    pooled_metrics[
                        "parkinson_recall"
                    ]
                ),
                "pooled_roc_auc": (
                    pooled_metrics["roc_auc"]
                ),
                "fold_macro_f1_mean": float(
                    current_folds[
                        "macro_f1"
                    ].mean()
                ),
                "fold_macro_f1_std": float(
                    current_folds[
                        "macro_f1"
                    ].std()
                ),
                "fold_balanced_accuracy_mean": float(
                    current_folds[
                        "balanced_accuracy"
                    ].mean()
                ),
                "fold_balanced_accuracy_std": float(
                    current_folds[
                        "balanced_accuracy"
                    ].std()
                ),
                "fold_roc_auc_mean": float(
                    current_folds[
                        "roc_auc"
                    ].mean()
                ),
                "fold_roc_auc_std": float(
                    current_folds[
                        "roc_auc"
                    ].std()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=[
            "fold_roc_auc_mean",
            "fold_macro_f1_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_activity_scores = (
        load_subject_activity_scores()
    )

    expected_rows = 355 * 11

    if len(subject_activity_scores) != expected_rows:
        raise RuntimeError(
            "Unexpected number of subject/activity rows. "
            f"Expected {expected_rows}, "
            f"found {len(subject_activity_scores)}."
        )

    subject_counts = (
        subject_activity_scores.groupby(
            "activity"
        )["subject_id"]
        .nunique()
    )

    if not np.all(subject_counts == 355):
        raise RuntimeError(
            "Not every activity has 355 subjects."
        )

    fold_metrics = calculate_fold_metrics(
        subject_activity_scores
    )

    activity_summary = (
        calculate_activity_summary(
            scores=subject_activity_scores,
            fold_metrics=fold_metrics,
        )
    )

    subject_activity_scores.to_csv(
        ACTIVITY_SCORES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    activity_summary.to_csv(
        ACTIVITY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 90)
    print("ACTIVITY-LEVEL SVM ANALYSIS")
    print("=" * 90)

    print(
        activity_summary[
            [
                "rank",
                "activity",
                "fold_roc_auc_mean",
                "fold_roc_auc_std",
                "fold_macro_f1_mean",
                "pooled_balanced_accuracy",
                "pooled_healthy_recall",
                "pooled_parkinson_recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nCreated files:")

    for path in [
        ACTIVITY_SCORES_PATH,
        FOLD_METRICS_PATH,
        ACTIVITY_SUMMARY_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()