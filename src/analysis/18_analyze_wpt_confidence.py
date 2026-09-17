from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "repeated_wpt_validation"
)

SUBJECT_STABILITY_PATH = (
    INPUT_DIR / "subject_stability.csv"
)

REPEAT_RESULTS_PATH = (
    INPUT_DIR / "repeat_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "wpt_selective_analysis"
)

THRESHOLD_TABLE_PATH = (
    OUTPUT_DIR / "confidence_threshold_metrics.csv"
)

COVERAGE_TABLE_PATH = (
    OUTPUT_DIR / "target_coverage_metrics.csv"
)

ERROR_STABILITY_PATH = (
    OUTPUT_DIR / "error_stability_summary.csv"
)

DIFFICULT_SUBJECTS_PATH = (
    OUTPUT_DIR / "persistent_difficult_subjects.csv"
)

RISK_COVERAGE_FIGURE_PATH = (
    OUTPUT_DIR / "risk_coverage_curve.png"
)

ACCURACY_COVERAGE_FIGURE_PATH = (
    OUTPUT_DIR / "accuracy_coverage_curve.png"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
]

TARGET_COVERAGES = [
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.60,
    0.50,
]

EXPECTED_SUBJECTS = 355
EXPECTED_REPEATS = 5


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    unique_classes = np.unique(y_true)

    if len(unique_classes) == 2:
        balanced_accuracy = float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

    else:
        balanced_accuracy = float("nan")
        macro_f1 = float("nan")
        roc_auc = float("nan")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_f1": macro_f1,
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
        "roc_auc": roc_auc,
        "healthy_correct": int(
            matrix[0, 0]
        ),
        "healthy_incorrect": int(
            matrix[0, 1]
        ),
        "parkinson_incorrect": int(
            matrix[1, 0]
        ),
        "parkinson_correct": int(
            matrix[1, 1]
        ),
    }


# ============================================================
# Data loading
# ============================================================

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not SUBJECT_STABILITY_PATH.exists():
        raise FileNotFoundError(
            "Subject-stability file not found: "
            f"{SUBJECT_STABILITY_PATH}"
        )

    if not REPEAT_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Repeat-results file not found: "
            f"{REPEAT_RESULTS_PATH}"
        )

    stability = pd.read_csv(
        SUBJECT_STABILITY_PATH,
        dtype={
            "subject_id": "string",
        },
    )

    repeat_results = pd.read_csv(
        REPEAT_RESULTS_PATH
    )

    required_stability_columns = {
        "subject_id",
        "y_true",
        "true_label",
        "n_predictions",
        "correct_count",
        "incorrect_count",
        "error_rate",
        "parkinson_prediction_rate",
        "mean_normalized_margin",
        "std_normalized_margin",
        "consensus_y_pred",
        "consensus_correct",
    }

    missing_columns = (
        required_stability_columns
        - set(stability.columns)
    )

    if missing_columns:
        raise ValueError(
            "Subject-stability file is missing: "
            f"{sorted(missing_columns)}"
        )

    if len(stability) != EXPECTED_SUBJECTS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_SUBJECTS} subjects, "
            f"found {len(stability)}."
        )

    if (
        stability["subject_id"].nunique()
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            "Subjects are duplicated or missing."
        )

    if not np.all(
        stability["n_predictions"]
        == EXPECTED_REPEATS
    ):
        raise ValueError(
            "Not every subject has exactly "
            f"{EXPECTED_REPEATS} predictions."
        )

    if len(repeat_results) != EXPECTED_REPEATS:
        raise ValueError(
            "Unexpected number of repeated "
            "validation results."
        )

    numeric_columns = [
        "y_true",
        "n_predictions",
        "correct_count",
        "incorrect_count",
        "error_rate",
        "parkinson_prediction_rate",
        "mean_normalized_margin",
        "std_normalized_margin",
        "consensus_y_pred",
    ]

    for column in numeric_columns:
        if not np.all(
            np.isfinite(
                stability[
                    column
                ].to_numpy(
                    dtype=float
                )
            )
        ):
            raise ValueError(
                f"Invalid values in {column}."
            )

    stability[
        "absolute_consensus_margin"
    ] = np.abs(
        stability[
            "mean_normalized_margin"
        ].to_numpy(dtype=float)
    )

    stability[
        "consensus_correct"
    ] = (
        stability[
            "consensus_y_pred"
        ].to_numpy(dtype=np.int8)
        == stability[
            "y_true"
        ].to_numpy(dtype=np.int8)
    )

    return stability, repeat_results


# ============================================================
# Selective evaluation
# ============================================================

def evaluate_subset(
    subset: pd.DataFrame,
    total_subjects: int,
    selection_rule: str,
    selection_value: float,
) -> dict[str, Any]:

    y_true = subset[
        "y_true"
    ].to_numpy(dtype=np.int8)

    y_pred = subset[
        "consensus_y_pred"
    ].to_numpy(dtype=np.int8)

    scores = subset[
        "mean_normalized_margin"
    ].to_numpy(dtype=float)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
    )

    retained_healthy = int(
        np.sum(y_true == 0)
    )

    retained_parkinson = int(
        np.sum(y_true == 1)
    )

    retained_subjects = len(subset)

    return {
        "selection_rule": selection_rule,
        "selection_value": (
            selection_value
        ),
        "retained_subjects": (
            retained_subjects
        ),
        "deferred_subjects": (
            total_subjects
            - retained_subjects
        ),
        "coverage": (
            retained_subjects
            / total_subjects
        ),
        "defer_rate": (
            1.0
            - retained_subjects
            / total_subjects
        ),
        "retained_healthy": (
            retained_healthy
        ),
        "retained_parkinson": (
            retained_parkinson
        ),
        "errors": int(
            np.sum(y_true != y_pred)
        ),
        "risk": float(
            np.mean(y_true != y_pred)
        ),
        **metrics,
    }


def build_threshold_table(
    stability: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[
        dict[str, Any]
    ] = []

    for threshold in CONFIDENCE_THRESHOLDS:

        subset = stability[
            stability[
                "absolute_consensus_margin"
            ] >= threshold
        ].copy()

        if subset.empty:
            continue

        rows.append(
            evaluate_subset(
                subset=subset,
                total_subjects=len(
                    stability
                ),
                selection_rule=(
                    "minimum_absolute_margin"
                ),
                selection_value=(
                    threshold
                ),
            )
        )

    return pd.DataFrame(rows)


def build_coverage_table(
    stability: pd.DataFrame,
) -> pd.DataFrame:

    ranked = (
        stability.sort_values(
            by=[
                "absolute_consensus_margin",
                "subject_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    rows: list[
        dict[str, Any]
    ] = []

    total_subjects = len(
        ranked
    )

    for target_coverage in (
        TARGET_COVERAGES
    ):

        retained_count = int(
            np.ceil(
                target_coverage
                * total_subjects
            )
        )

        retained_count = min(
            max(
                retained_count,
                1,
            ),
            total_subjects,
        )

        subset = ranked.iloc[
            :retained_count
        ].copy()

        minimum_margin = float(
            subset[
                "absolute_consensus_margin"
            ].min()
        )

        row = evaluate_subset(
            subset=subset,
            total_subjects=(
                total_subjects
            ),
            selection_rule=(
                "target_coverage"
            ),
            selection_value=(
                target_coverage
            ),
        )

        row[
            "actual_minimum_absolute_margin"
        ] = minimum_margin

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Error stability
# ============================================================

def build_error_stability_summary(
    stability: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        stability.groupby(
            [
                "true_label",
                "incorrect_count",
            ],
            as_index=False,
        )
        .agg(
            subjects=(
                "subject_id",
                "size",
            ),
            mean_absolute_margin=(
                "absolute_consensus_margin",
                "mean",
            ),
            mean_margin_variability=(
                "std_normalized_margin",
                "mean",
            ),
        )
    )

    class_totals = (
        stability.groupby(
            "true_label"
        )["subject_id"]
        .size()
        .to_dict()
    )

    summary[
        "within_class_rate"
    ] = [
        row.subjects
        / class_totals[
            row.true_label
        ]
        for row
        in summary.itertuples(
            index=False
        )
    ]

    return summary


def build_difficult_subject_table(
    stability: pd.DataFrame,
) -> pd.DataFrame:

    difficult = stability[
        stability[
            "incorrect_count"
        ] >= 3
    ].copy()

    difficult[
        "difficulty_group"
    ] = np.select(
        [
            difficult[
                "incorrect_count"
            ] == EXPECTED_REPEATS,
            difficult[
                "incorrect_count"
            ] == (
                EXPECTED_REPEATS - 1
            ),
        ],
        [
            "always_incorrect",
            "incorrect_in_four_repeats",
        ],
        default=(
            "incorrect_in_three_repeats"
        ),
    )

    difficult[
        "confidently_wrong"
    ] = (
        (
            difficult[
                "consensus_correct"
            ] == False
        )
        & (
            difficult[
                "absolute_consensus_margin"
            ] >= 0.50
        )
    )

    difficult = (
        difficult.sort_values(
            by=[
                "incorrect_count",
                "absolute_consensus_margin",
                "std_normalized_margin",
                "subject_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    selected_columns = [
        "subject_id",
        "true_label",
        "incorrect_count",
        "error_rate",
        "difficulty_group",
        "consensus_y_pred",
        "consensus_correct",
        "confidently_wrong",
        "parkinson_prediction_rate",
        "mean_normalized_margin",
        "absolute_consensus_margin",
        "std_normalized_margin",
        "minimum_normalized_margin",
        "maximum_normalized_margin",
    ]

    return difficult[
        selected_columns
    ]


# ============================================================
# Figures
# ============================================================

def save_curves(
    threshold_table: pd.DataFrame,
) -> None:

    ordered = (
        threshold_table.sort_values(
            "coverage"
        )
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 5.0)
    )

    axis.plot(
        ordered["coverage"],
        ordered["risk"],
        marker="o",
    )

    axis.set_xlabel(
        "Coverage"
    )

    axis.set_ylabel(
        "Error rate (risk)"
    )

    axis.set_title(
        "WPT-SVM selective risk–coverage curve"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        RISK_COVERAGE_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7.0, 5.0)
    )

    axis.plot(
        ordered["coverage"],
        ordered["accuracy"],
        marker="o",
    )

    axis.set_xlabel(
        "Coverage"
    )

    axis.set_ylabel(
        "Accuracy on retained subjects"
    )

    axis.set_title(
        "WPT-SVM accuracy–coverage curve"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        ACCURACY_COVERAGE_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        stability,
        repeat_results,
    ) = load_data()

    threshold_table = (
        build_threshold_table(
            stability
        )
    )

    coverage_table = (
        build_coverage_table(
            stability
        )
    )

    error_summary = (
        build_error_stability_summary(
            stability
        )
    )

    difficult_subjects = (
        build_difficult_subject_table(
            stability
        )
    )

    save_curves(
        threshold_table
    )

    threshold_table.to_csv(
        THRESHOLD_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coverage_table.to_csv(
        COVERAGE_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    error_summary.to_csv(
        ERROR_STABILITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    difficult_subjects.to_csv(
        DIFFICULT_SUBJECTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    always_correct = int(
        np.sum(
            stability[
                "incorrect_count"
            ] == 0
        )
    )

    always_incorrect = int(
        np.sum(
            stability[
                "incorrect_count"
            ] == EXPECTED_REPEATS
        )
    )

    incorrect_three_or_more = int(
        np.sum(
            stability[
                "incorrect_count"
            ] >= 3
        )
    )

    confident_wrong = int(
        np.sum(
            (
                stability[
                    "consensus_correct"
                ] == False
            )
            & (
                stability[
                    "absolute_consensus_margin"
                ] >= 0.50
            )
        )
    )

    print("=" * 100)
    print("WPT-SVM CONFIDENCE AND SELECTIVE-CLASSIFICATION ANALYSIS")
    print("=" * 100)

    print(
        f"\nSubjects: {len(stability)}"
    )

    print(
        "Repeated validation runs: "
        f"{len(repeat_results)}"
    )

    print(
        "Always correctly classified: "
        f"{always_correct}"
    )

    print(
        "Always incorrectly classified: "
        f"{always_incorrect}"
    )

    print(
        "Incorrect in at least 3/5 repeats: "
        f"{incorrect_three_or_more}"
    )

    print(
        "Consensus errors with "
        "|mean margin| >= 0.50: "
        f"{confident_wrong}"
    )

    print("\nConfidence-threshold results:")

    print(
        threshold_table[
            [
                "selection_value",
                "retained_subjects",
                "coverage",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "healthy_recall",
                "parkinson_recall",
                "errors",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nTarget-coverage results:")

    print(
        coverage_table[
            [
                "selection_value",
                "retained_subjects",
                "coverage",
                "actual_minimum_absolute_margin",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "errors",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nPersistent difficult subjects:")

    print(
        difficult_subjects.head(
            25
        ).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nCreated files:")

    for output_path in [
        THRESHOLD_TABLE_PATH,
        COVERAGE_TABLE_PATH,
        ERROR_STABILITY_PATH,
        DIFFICULT_SUBJECTS_PATH,
        RISK_COVERAGE_FIGURE_PATH,
        ACCURACY_COVERAGE_FIGURE_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
