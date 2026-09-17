from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.svm import SVC


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_PATH = (
    OUTPUT_DIR / "dwt_features_db4_level5.parquet"
)

RESULTS_DIR = OUTPUT_DIR / "baseline_results"
FIGURES_DIR = RESULTS_DIR / "figures"

FOLD_METRICS_PATH = (
    RESULTS_DIR / "baseline_fold_metrics.csv"
)

OVERALL_METRICS_PATH = (
    RESULTS_DIR / "baseline_overall_metrics.csv"
)

FOLD_SUMMARY_PATH = (
    RESULTS_DIR / "baseline_fold_summary.csv"
)

WINDOW_PREDICTIONS_PATH = (
    RESULTS_DIR / "baseline_window_predictions.parquet"
)

SUBJECT_PREDICTIONS_PATH = (
    RESULTS_DIR / "baseline_subject_predictions.csv"
)

SELECTED_FEATURES_PATH = (
    RESULTS_DIR / "selected_feature_frequency.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

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
# Models
# ============================================================

def create_models() -> dict[str, Pipeline]:
    """
    Fixed baseline models.

    Hyperparameter tuning is intentionally postponed until
    after obtaining an honest baseline.
    """

    svm_model = Pipeline(
        steps=[
            (
                "variance",
                VarianceThreshold(threshold=0.0),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=200,
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
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=1024,
                ),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "variance",
                VarianceThreshold(threshold=0.0),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=300,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "SVM_RBF": svm_model,
        "Random_Forest": random_forest_model,
    }


# ============================================================
# Data validation
# ============================================================

def load_dataset() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    data = pd.read_parquet(FEATURE_PATH)

    missing_meta_columns = (
        set(META_COLUMNS) - set(data.columns)
    )

    if missing_meta_columns:
        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_meta_columns)}"
        )

    data["subject_id"] = (
        data["subject_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(3)
    )

    unknown_labels = (
        set(data["label"].unique())
        - set(LABEL_TO_ID)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in META_COLUMNS
    ]

    if len(feature_columns) != 648:
        raise ValueError(
            "Unexpected feature count: "
            f"{len(feature_columns)}"
        )

    label_count_per_subject = (
        data.groupby("subject_id")["label"].nunique()
    )

    invalid_subjects = label_count_per_subject[
        label_count_per_subject != 1
    ]

    if not invalid_subjects.empty:
        raise ValueError(
            "Some subjects have multiple labels: "
            f"{invalid_subjects.index.tolist()}"
        )

    windows_per_subject = (
        data.groupby("subject_id").size()
    )

    if not np.all(windows_per_subject == 45):
        raise ValueError(
            "Not every subject has exactly 45 windows."
        )

    X = data[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = (
        data["label"]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    groups = data["subject_id"].to_numpy()

    if not np.all(np.isfinite(X)):
        invalid_count = int(
            X.size - np.isfinite(X).sum()
        )

        raise ValueError(
            f"Feature matrix contains "
            f"{invalid_count} invalid values."
        )

    return data, X, y, groups, feature_columns


# ============================================================
# Prediction scores
# ============================================================

def get_prediction_scores(
    model: Pipeline,
    X: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Returns:
        scores
        default classification threshold

    SVM decision function:
        threshold = 0

    Random Forest probability:
        threshold = 0.5
    """

    if hasattr(model, "decision_function"):
        scores = np.asarray(
            model.decision_function(X)
        ).reshape(-1)

        return scores, 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        scores = probabilities[:, 1]

        return scores, 0.5

    predictions = model.predict(X)

    return (
        np.asarray(predictions, dtype=float),
        0.5,
    )


# ============================================================
# Subject-level aggregation
# ============================================================

def aggregate_subject_predictions(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Step 1:
        Average windows within each activity.

    Step 2:
        Average the 11 activity scores for each subject.

    This gives every activity equal weight.
    """

    temporary = metadata[
        [
            "subject_id",
            "activity",
        ]
    ].copy()

    temporary["y_true"] = y_true
    temporary["score"] = scores

    activity_label_counts = (
        temporary.groupby(
            ["subject_id", "activity"]
        )["y_true"]
        .nunique()
    )

    if not np.all(activity_label_counts == 1):
        raise ValueError(
            "Inconsistent labels within subject/activity."
        )

    activity_predictions = (
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

    subject_label_counts = (
        activity_predictions.groupby(
            "subject_id"
        )["y_true"]
        .nunique()
    )

    if not np.all(subject_label_counts == 1):
        raise ValueError(
            "Inconsistent labels at subject level."
        )

    subject_predictions = (
        activity_predictions.groupby(
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

    if not np.all(
        subject_predictions["n_activities"] == 11
    ):
        raise ValueError(
            "Not every subject has 11 activities."
        )

    subject_predictions["y_pred"] = (
        subject_predictions["score"] >= threshold
    ).astype(np.int8)

    subject_predictions["true_label"] = (
        subject_predictions["y_true"]
        .map(ID_TO_LABEL)
    )

    subject_predictions["predicted_label"] = (
        subject_predictions["y_pred"]
        .map(ID_TO_LABEL)
    )

    return subject_predictions


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "healthy_precision": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            average="binary",
            zero_division=0,
        ),
        "healthy_recall": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            average="binary",
            zero_division=0,
        ),
        "healthy_f1": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            average="binary",
            zero_division=0,
        ),
        "parkinson_precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "parkinson_recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "parkinson_f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
    }

    if (
        scores is not None
        and len(np.unique(y_true)) == 2
    ):
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            scores,
        )
    else:
        metrics["roc_auc"] = np.nan

    return {
        key: float(value)
        for key, value in metrics.items()
    }


# ============================================================
# Selected-feature tracking
# ============================================================

def get_selected_features(
    fitted_model: Pipeline,
    original_feature_columns: list[str],
) -> list[str]:
    variance_selector = (
        fitted_model.named_steps["variance"]
    )

    selection_model = (
        fitted_model.named_steps["select"]
    )

    variance_mask = (
        variance_selector.get_support()
    )

    after_variance = np.asarray(
        original_feature_columns
    )[variance_mask]

    selection_mask = (
        selection_model.get_support()
    )

    return after_variance[
        selection_mask
    ].tolist()


# ============================================================
# Saving figures and reports
# ============================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
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
        f"{model_name} — Subject-level OOF"
    )

    figure.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    figure.savefig(
        FIGURES_DIR
        / f"{safe_name}_subject_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
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

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    report_path = (
        RESULTS_DIR
        / f"{safe_name}_subject_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# Main evaluation
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
        data,
        X,
        y,
        groups,
        feature_columns,
    ) = load_dataset()

    models = create_models()

    cross_validator = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics: list[dict[str, Any]] = []

    all_window_predictions: list[pd.DataFrame] = []
    all_subject_predictions: list[pd.DataFrame] = []

    feature_counters = {
        model_name: Counter()
        for model_name in models
    }

    print("=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Subjects: "
        f"{data['subject_id'].nunique()}"
    )
    print(f"Cross-validation folds: {N_SPLITS}")

    for model_name, model_template in models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        split_iterator = cross_validator.split(
            X=X,
            y=y,
            groups=groups,
        )

        for fold_number, (
            train_indices,
            test_indices,
        ) in enumerate(
            split_iterator,
            start=1,
        ):
            train_subjects = set(
                groups[train_indices]
            )

            test_subjects = set(
                groups[test_indices]
            )

            overlap = (
                train_subjects
                .intersection(test_subjects)
            )

            if overlap:
                raise RuntimeError(
                    "Subject leakage detected: "
                    f"{sorted(overlap)}"
                )

            print(
                f"\nFold {fold_number}/{N_SPLITS}"
            )

            print(
                "Train subjects: "
                f"{len(train_subjects)}"
            )

            print(
                "Test subjects: "
                f"{len(test_subjects)}"
            )

            fitted_model = clone(model_template)

            fitted_model.fit(
                X[train_indices],
                y[train_indices],
            )

            window_predictions = (
                fitted_model.predict(
                    X[test_indices]
                )
            )

            window_scores, threshold = (
                get_prediction_scores(
                    fitted_model,
                    X[test_indices],
                )
            )

            window_metric_values = (
                calculate_metrics(
                    y_true=y[test_indices],
                    y_pred=window_predictions,
                    scores=window_scores,
                )
            )

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "level": "window",
                    "n_samples": len(test_indices),
                    "n_subjects": len(test_subjects),
                    **window_metric_values,
                }
            )

            window_frame = data.iloc[
                test_indices
            ][
                [
                    "subject_id",
                    "activity",
                    "window_id",
                    "start_index",
                    "end_index",
                ]
            ].copy()

            window_frame["model"] = model_name
            window_frame["fold"] = fold_number
            window_frame["y_true"] = y[test_indices]
            window_frame["y_pred"] = (
                window_predictions
            )
            window_frame["score"] = window_scores
            window_frame["true_label"] = (
                window_frame["y_true"]
                .map(ID_TO_LABEL)
            )
            window_frame["predicted_label"] = (
                window_frame["y_pred"]
                .map(ID_TO_LABEL)
            )

            all_window_predictions.append(
                window_frame
            )

            subject_frame = (
                aggregate_subject_predictions(
                    metadata=data.iloc[test_indices],
                    y_true=y[test_indices],
                    scores=window_scores,
                    threshold=threshold,
                )
            )

            subject_frame["model"] = model_name
            subject_frame["fold"] = fold_number
            subject_frame["threshold"] = threshold

            subject_metric_values = (
                calculate_metrics(
                    y_true=subject_frame[
                        "y_true"
                    ].to_numpy(),
                    y_pred=subject_frame[
                        "y_pred"
                    ].to_numpy(),
                    scores=subject_frame[
                        "score"
                    ].to_numpy(),
                )
            )

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "level": "subject",
                    "n_samples": len(subject_frame),
                    "n_subjects": len(subject_frame),
                    **subject_metric_values,
                }
            )

            all_subject_predictions.append(
                subject_frame
            )

            selected_features = (
                get_selected_features(
                    fitted_model=fitted_model,
                    original_feature_columns=(
                        feature_columns
                    ),
                )
            )

            feature_counters[
                model_name
            ].update(selected_features)

            print(
                "Window Macro-F1: "
                f"{window_metric_values['macro_f1']:.4f}"
            )

            print(
                "Subject Accuracy: "
                f"{subject_metric_values['accuracy']:.4f}"
            )

            print(
                "Subject Macro-F1: "
                f"{subject_metric_values['macro_f1']:.4f}"
            )

            print(
                "Healthy Recall: "
                f"{subject_metric_values['healthy_recall']:.4f}"
            )

            print(
                "Parkinson Recall: "
                f"{subject_metric_values['parkinson_recall']:.4f}"
            )

    # ========================================================
    # Combine predictions
    # ========================================================

    fold_metrics_table = pd.DataFrame(
        fold_metrics
    )

    window_predictions_table = pd.concat(
        all_window_predictions,
        ignore_index=True,
    )

    subject_predictions_table = pd.concat(
        all_subject_predictions,
        ignore_index=True,
    )

    # Every model must predict every original row once
    expected_window_rows = (
        len(data) * len(models)
    )

    if (
        len(window_predictions_table)
        != expected_window_rows
    ):
        raise RuntimeError(
            "Unexpected number of out-of-fold "
            "window predictions."
        )

    expected_subject_rows = (
        data["subject_id"].nunique()
        * len(models)
    )

    if (
        len(subject_predictions_table)
        != expected_subject_rows
    ):
        raise RuntimeError(
            "Unexpected number of out-of-fold "
            "subject predictions."
        )

    # ========================================================
    # Overall out-of-fold metrics
    # ========================================================

    overall_rows: list[dict[str, Any]] = []

    for model_name in models:
        model_windows = (
            window_predictions_table[
                window_predictions_table[
                    "model"
                ] == model_name
            ]
        )

        window_metrics = calculate_metrics(
            y_true=model_windows[
                "y_true"
            ].to_numpy(),
            y_pred=model_windows[
                "y_pred"
            ].to_numpy(),
            scores=model_windows[
                "score"
            ].to_numpy(),
        )

        overall_rows.append(
            {
                "model": model_name,
                "level": "window",
                "n_samples": len(model_windows),
                "n_subjects": (
                    model_windows[
                        "subject_id"
                    ].nunique()
                ),
                **window_metrics,
            }
        )

        model_subjects = (
            subject_predictions_table[
                subject_predictions_table[
                    "model"
                ] == model_name
            ]
        )

        subject_metrics = calculate_metrics(
            y_true=model_subjects[
                "y_true"
            ].to_numpy(),
            y_pred=model_subjects[
                "y_pred"
            ].to_numpy(),
            scores=model_subjects[
                "score"
            ].to_numpy(),
        )

        overall_rows.append(
            {
                "model": model_name,
                "level": "subject",
                "n_samples": len(model_subjects),
                "n_subjects": len(model_subjects),
                **subject_metrics,
            }
        )

        save_confusion_matrix(
            y_true=model_subjects[
                "y_true"
            ].to_numpy(),
            y_pred=model_subjects[
                "y_pred"
            ].to_numpy(),
            model_name=model_name,
        )

        save_classification_report(
            y_true=model_subjects[
                "y_true"
            ].to_numpy(),
            y_pred=model_subjects[
                "y_pred"
            ].to_numpy(),
            model_name=model_name,
        )

    overall_metrics_table = pd.DataFrame(
        overall_rows
    )

    # ========================================================
    # Fold mean and standard deviation
    # ========================================================

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "healthy_precision",
        "healthy_recall",
        "healthy_f1",
        "parkinson_precision",
        "parkinson_recall",
        "parkinson_f1",
        "roc_auc",
    ]

    fold_summary = (
        fold_metrics_table.groupby(
            ["model", "level"]
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    fold_summary.columns = [
        (
            column
            if isinstance(column, str)
            else "_".join(
                part
                for part in column
                if part
            )
        )
        for column in fold_summary.columns
    ]

    # ========================================================
    # Selected-feature frequency
    # ========================================================

    selected_feature_rows = []

    for model_name, counter in (
        feature_counters.items()
    ):
        for feature_name, count in (
            counter.most_common()
        ):
            selected_feature_rows.append(
                {
                    "model": model_name,
                    "feature_name": feature_name,
                    "selected_folds": count,
                    "selection_rate": (
                        count / N_SPLITS
                    ),
                }
            )

    selected_features_table = pd.DataFrame(
        selected_feature_rows
    )

    # ========================================================
    # Save results
    # ========================================================

    fold_metrics_table.to_csv(
        FOLD_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics_table.to_csv(
        OVERALL_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary.to_csv(
        FOLD_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    window_predictions_table.to_parquet(
        WINDOW_PREDICTIONS_PATH,
        index=False,
    )

    subject_predictions_table.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    selected_features_table.to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Final console summary
    # ========================================================

    subject_summary = (
        overall_metrics_table[
            overall_metrics_table[
                "level"
            ] == "subject"
        ]
        .sort_values(
            "macro_f1",
            ascending=False,
        )
    )

    print("\n" + "=" * 70)
    print("OVERALL SUBJECT-LEVEL RESULTS")
    print("=" * 70)

    print(
        subject_summary[
            [
                "model",
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

    print("\nCreated result files:")

    for path in [
        FOLD_METRICS_PATH,
        OVERALL_METRICS_PATH,
        FOLD_SUMMARY_PATH,
        WINDOW_PREDICTIONS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        SELECTED_FEATURES_PATH,
        FIGURES_DIR,
    ]:
        print(path)


if __name__ == "__main__":
    main()