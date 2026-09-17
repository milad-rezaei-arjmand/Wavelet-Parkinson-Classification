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
from sklearn.svm import SVC
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

WPT_FEATURE_PATH = (
    OUTPUTS_DIR
    / "wpt_features_db4_level4.parquet"
)

RESULTS_DIR = (
    OUTPUTS_DIR
    / "wpt_svm_tuning"
)

FIGURES_DIR = RESULTS_DIR / "figures"

FOLD_RESULTS_PATH = (
    RESULTS_DIR / "fold_results.csv"
)

SEARCH_RESULTS_PATH = (
    RESULTS_DIR / "configuration_search.csv"
)

SUBJECT_PREDICTIONS_PATH = (
    RESULTS_DIR / "subject_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIR / "overall_summary.csv"
)

SELECTED_FEATURES_PATH = (
    RESULTS_DIR / "selected_feature_frequency.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "wpt_tuned_svm_confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = (
    RESULTS_DIR / "classification_report.txt"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

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
# Candidate configurations
# ============================================================

def create_candidate_configurations() -> list[
    dict[str, Any]
]:
    """
    Compact coarse search.

    Total:
        5 baseline feature-count configurations
        6 alternative-C configurations
        6 alternative-gamma configurations
        = 17 configurations
    """

    configurations: list[
        dict[str, Any]
    ] = []

    # Feature-count screening with baseline SVM settings
    for selected_features in [
        200,
        300,
        500,
        800,
        1200,
    ]:
        configurations.append(
            {
                "selected_features": selected_features,
                "C": 10.0,
                "gamma": "scale",
            }
        )

    # C screening
    for selected_features in [
        300,
        500,
        800,
    ]:
        for c_value in [
            1.0,
            30.0,
        ]:
            configurations.append(
                {
                    "selected_features": selected_features,
                    "C": c_value,
                    "gamma": "scale",
                }
            )

    # Gamma screening
    for selected_features in [
        300,
        500,
        800,
    ]:
        for gamma_value in [
            0.001,
            0.003,
        ]:
            configurations.append(
                {
                    "selected_features": selected_features,
                    "C": 10.0,
                    "gamma": gamma_value,
                }
            )

    unique_configurations: list[
        dict[str, Any]
    ] = []

    observed = set()

    for configuration in configurations:
        key = (
            configuration["selected_features"],
            configuration["C"],
            str(configuration["gamma"]),
        )

        if key not in observed:
            observed.add(key)
            unique_configurations.append(
                configuration
            )

    for configuration_id, configuration in enumerate(
        unique_configurations,
        start=1,
    ):
        configuration["configuration_id"] = (
            configuration_id
        )

    return unique_configurations


CANDIDATE_CONFIGURATIONS = (
    create_candidate_configurations()
)


# ============================================================
# Data loading
# ============================================================

def load_wpt_dataset() -> tuple[
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

    data["subject_id"] = (
        data["subject_id"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
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
            "Unexpected WPT row count: "
            f"{len(data)}"
        )

    if data["subject_id"].nunique() != 355:
        raise ValueError(
            "Expected 355 subjects."
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

    print("Converting features to NumPy...")

    X = data[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.all(np.isfinite(X)):
        invalid_count = int(
            X.size
            - np.isfinite(X).sum()
        )

        raise ValueError(
            f"WPT matrix contains "
            f"{invalid_count} invalid values."
        )

    y = (
        data["label"]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=np.int8)
    )

    groups = (
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
        X,
        y,
        groups,
        feature_columns,
    )


# ============================================================
# Model
# ============================================================

def create_model(
    configuration: dict[str, Any],
) -> Pipeline:

    return Pipeline(
        steps=[
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
                    k=int(
                        configuration[
                            "selected_features"
                        ]
                    ),
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
                    C=float(
                        configuration["C"]
                    ),
                    gamma=configuration[
                        "gamma"
                    ],
                    class_weight="balanced",
                    cache_size=1024,
                ),
            ),
        ]
    )


def get_decision_scores(
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
    """
    Windows are averaged inside each activity first.

    Then the 11 activity scores are averaged so that
    2048-sample activities do not receive extra weight.
    """

    temporary = metadata[
        [
            "subject_id",
            "activity",
        ]
    ].copy()

    temporary["y_true"] = np.asarray(
        y_true,
        dtype=np.int8,
    )

    temporary["score"] = np.asarray(
        scores,
        dtype=float,
    )

    label_counts = (
        temporary.groupby(
            [
                "subject_id",
                "activity",
            ]
        )["y_true"]
        .nunique()
    )

    if not np.all(label_counts == 1):
        raise ValueError(
            "Inconsistent labels inside "
            "subject/activity groups."
        )

    activity_scores = (
        temporary.groupby(
            [
                "subject_id",
                "activity",
            ],
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
            score=(
                "activity_score",
                "mean",
            ),
            n_activities=(
                "activity",
                "nunique",
            ),
            n_windows=(
                "n_windows",
                "sum",
            ),
        )
    )

    if not np.all(
        subject_scores[
            "n_activities"
        ] == 11
    ):
        raise ValueError(
            "Not every subject has 11 activities."
        )

    if not np.all(
        subject_scores[
            "n_windows"
        ] == 45
    ):
        raise ValueError(
            "Not every subject has 45 windows."
        )

    return subject_scores


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

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


# ============================================================
# Threshold selection
# ============================================================

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
            "No scores available for threshold tuning."
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

    candidates = np.concatenate(
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

    return np.unique(candidates)


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
# Inner evaluation
# ============================================================

def evaluate_configuration_inner(
    configuration: dict[str, Any],
    inner_splits_relative: list[
        tuple[np.ndarray, np.ndarray]
    ],
    outer_train_indices: np.ndarray,
    metadata: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:

    subject_frames: list[
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

        train_subjects = set(
            groups[
                inner_train_indices
            ]
        )

        validation_subjects = set(
            groups[
                inner_validation_indices
            ]
        )

        if train_subjects.intersection(
            validation_subjects
        ):
            raise RuntimeError(
                "Subject leakage inside inner CV."
            )

        model = create_model(
            configuration
        )

        model.fit(
            X[inner_train_indices],
            y[inner_train_indices],
        )

        validation_scores = (
            get_decision_scores(
                fitted_model=model,
                X=X[
                    inner_validation_indices
                ],
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
                scores=validation_scores,
            )
        )

        subject_frames.append(
            validation_subject_scores
        )

        del model
        gc.collect()

    inner_predictions = pd.concat(
        subject_frames,
        ignore_index=True,
    )

    expected_subjects = set(
        groups[
            outer_train_indices
        ]
    )

    actual_subjects = set(
        inner_predictions[
            "subject_id"
        ]
    )

    if actual_subjects != expected_subjects:
        missing_subjects = sorted(
            expected_subjects
            - actual_subjects
        )

        raise RuntimeError(
            "Incomplete inner OOF predictions. "
            f"Missing: {missing_subjects}"
        )

    subject_counts = (
        inner_predictions[
            "subject_id"
        ].value_counts()
    )

    if not np.all(
        subject_counts == 1
    ):
        raise RuntimeError(
            "Subjects duplicated in inner OOF."
        )

    inner_y_true = (
        inner_predictions[
            "y_true"
        ].to_numpy(
            dtype=np.int8
        )
    )

    inner_scores = (
        inner_predictions[
            "score"
        ].to_numpy(
            dtype=float
        )
    )

    threshold, metrics = (
        find_best_threshold(
            y_true=inner_y_true,
            scores=inner_scores,
        )
    )

    score_std = float(
        np.std(inner_scores)
    )

    if score_std <= EPSILON:
        score_std = 1.0

    return {
        "threshold": threshold,
        "metrics": metrics,
        "score_std": score_std,
        "inner_scores": inner_scores,
        "inner_y_true": inner_y_true,
    }


# ============================================================
# Selected feature names
# ============================================================

def get_selected_feature_names(
    fitted_model: Pipeline,
    feature_columns: list[str],
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
        feature_columns
    )[
        variance_selector.get_support()
    ]

    selected_features = (
        after_variance[
            feature_selector.get_support()
        ]
    )

    return selected_features.tolist()


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
        metadata,
        X,
        y,
        groups,
        feature_columns,
    ) = load_wpt_dataset()

    print("\n" + "=" * 78)
    print("NESTED WPT-SVM HYPERPARAMETER TUNING")
    print("=" * 78)

    print(f"\nRows: {len(metadata)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Subjects: "
        f"{metadata['subject_id'].nunique()}"
    )
    print(
        "Candidate configurations: "
        f"{len(CANDIDATE_CONFIGURATIONS)}"
    )
    print(f"Outer folds: {N_OUTER_SPLITS}")
    print(f"Inner folds: {N_INNER_SPLITS}")

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    dummy_X = np.zeros(
        (
            len(y),
            1,
        ),
        dtype=np.float32,
    )

    outer_splits = list(
        outer_cv.split(
            X=dummy_X,
            y=y,
            groups=groups,
        )
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    search_rows: list[
        dict[str, Any]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    feature_counter: Counter[str] = Counter()

    for outer_fold, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        print("\n" + "=" * 78)
        print(
            f"OUTER FOLD "
            f"{outer_fold}/{N_OUTER_SPLITS}"
        )
        print("=" * 78)

        train_subjects = set(
            groups[
                outer_train_indices
            ]
        )

        test_subjects = set(
            groups[
                outer_test_indices
            ]
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

        inner_cv = StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=(
                RANDOM_STATE
                + outer_fold
            ),
        )

        inner_splits_relative = list(
            inner_cv.split(
                X=np.zeros(
                    (
                        len(
                            outer_train_indices
                        ),
                        1,
                    ),
                    dtype=np.float32,
                ),
                y=y[
                    outer_train_indices
                ],
                groups=groups[
                    outer_train_indices
                ],
            )
        )

        best_configuration: dict[
            str,
            Any
        ] | None = None

        best_inner_result: dict[
            str,
            Any
        ] | None = None

        best_key: tuple[
            float,
            ...
        ] | None = None

        progress = tqdm(
            CANDIDATE_CONFIGURATIONS,
            desc=(
                f"Outer fold {outer_fold} "
                "configurations"
            ),
            unit="config",
        )

        for configuration in progress:

            inner_result = (
                evaluate_configuration_inner(
                    configuration=configuration,
                    inner_splits_relative=(
                        inner_splits_relative
                    ),
                    outer_train_indices=(
                        outer_train_indices
                    ),
                    metadata=metadata,
                    X=X,
                    y=y,
                    groups=groups,
                )
            )

            metrics = inner_result[
                "metrics"
            ]

            minimum_class_recall = min(
                metrics["healthy_recall"],
                metrics[
                    "parkinson_recall"
                ],
            )

            selection_key = (
                metrics["macro_f1"],
                minimum_class_recall,
                metrics[
                    "balanced_accuracy"
                ],
                metrics["roc_auc"],
                -float(
                    configuration[
                        "selected_features"
                    ]
                ),
                -float(
                    configuration["C"]
                ),
            )

            search_rows.append(
                {
                    "outer_fold": outer_fold,
                    **configuration,
                    "selected_threshold": (
                        inner_result[
                            "threshold"
                        ]
                    ),
                    "inner_score_std": (
                        inner_result[
                            "score_std"
                        ]
                    ),
                    **{
                        f"inner_{key}": value
                        for key, value
                        in metrics.items()
                    },
                }
            )

            if (
                best_key is None
                or selection_key > best_key
            ):
                best_key = selection_key

                best_configuration = dict(
                    configuration
                )

                best_inner_result = (
                    inner_result
                )

            progress.set_postfix(
                best_macro_f1=(
                    "None"
                    if best_inner_result is None
                    else (
                        f"{best_inner_result['metrics']['macro_f1']:.4f}"
                    )
                )
            )

        if (
            best_configuration is None
            or best_inner_result is None
        ):
            raise RuntimeError(
                "No WPT configuration selected."
            )

        selected_threshold = float(
            best_inner_result[
                "threshold"
            ]
        )

        inner_score_std = float(
            best_inner_result[
                "score_std"
            ]
        )

        inner_metrics = (
            best_inner_result[
                "metrics"
            ]
        )

        print("\nSelected configuration:")

        print(
            "Features: "
            f"{best_configuration['selected_features']}"
        )

        print(
            f"C: {best_configuration['C']}"
        )

        print(
            f"Gamma: "
            f"{best_configuration['gamma']}"
        )

        print(
            "Threshold: "
            f"{selected_threshold:.6f}"
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

        final_model = create_model(
            best_configuration
        )

        final_model.fit(
            X[outer_train_indices],
            y[outer_train_indices],
        )

        outer_window_scores = (
            get_decision_scores(
                fitted_model=final_model,
                X=X[
                    outer_test_indices
                ],
            )
        )

        outer_subjects = (
            aggregate_subject_scores(
                metadata=metadata.iloc[
                    outer_test_indices
                ],
                y_true=y[
                    outer_test_indices
                ],
                scores=outer_window_scores,
            )
        )

        outer_subjects["y_pred"] = (
            outer_subjects["score"]
            >= selected_threshold
        ).astype(np.int8)

        outer_subjects[
            "normalized_margin"
        ] = (
            (
                outer_subjects["score"]
                - selected_threshold
            )
            / inner_score_std
        )

        outer_metrics = calculate_metrics(
            y_true=outer_subjects[
                "y_true"
            ].to_numpy(
                dtype=np.int8
            ),
            y_pred=outer_subjects[
                "y_pred"
            ].to_numpy(
                dtype=np.int8
            ),
            scores=outer_subjects[
                "score"
            ].to_numpy(
                dtype=float
            ),
        )

        selected_features = (
            get_selected_feature_names(
                fitted_model=final_model,
                feature_columns=feature_columns,
            )
        )

        feature_counter.update(
            selected_features
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
                **best_configuration,
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

        outer_subjects[
            "outer_fold"
        ] = outer_fold

        outer_subjects[
            "selected_features"
        ] = best_configuration[
            "selected_features"
        ]

        outer_subjects["C"] = (
            best_configuration["C"]
        )

        outer_subjects["gamma"] = str(
            best_configuration["gamma"]
        )

        outer_subjects[
            "selected_threshold"
        ] = selected_threshold

        outer_subjects[
            "true_label"
        ] = outer_subjects[
            "y_true"
        ].map(ID_TO_LABEL)

        outer_subjects[
            "predicted_label"
        ] = outer_subjects[
            "y_pred"
        ].map(ID_TO_LABEL)

        prediction_frames.append(
            outer_subjects
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

        del final_model
        gc.collect()

    # ========================================================
    # Overall metrics
    # ========================================================

    fold_table = pd.DataFrame(
        fold_rows
    )

    search_table = pd.DataFrame(
        search_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if len(predictions) != 355:
        raise RuntimeError(
            "Expected exactly 355 "
            "subject-level predictions."
        )

    if (
        predictions[
            "subject_id"
        ].nunique()
        != 355
    ):
        raise RuntimeError(
            "Subjects duplicated or missing."
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

    overall_metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=normalized_scores,
    )

    summary = pd.DataFrame(
        [
            {
                "method": (
                    "nested_tuned_wpt_svm"
                ),
                "n_subjects": len(
                    predictions
                ),
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
            }
        ]
    )

    selected_feature_rows = [
        {
            "feature_name": feature_name,
            "selected_folds": selected_folds,
            "selection_rate": (
                selected_folds
                / N_OUTER_SPLITS
            ),
        }
        for (
            feature_name,
            selected_folds,
        ) in feature_counter.most_common()
    ]

    selected_feature_table = (
        pd.DataFrame(
            selected_feature_rows
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
        "Nested tuned WPT-SVM"
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
    # Save files
    # ========================================================

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    search_table.to_csv(
        SEARCH_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    selected_feature_table.to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n" + "=" * 82)
    print("OVERALL TUNED WPT-SVM RESULTS")
    print("=" * 82)

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

    print("\nSelected configurations by fold:")

    print(
        fold_table[
            [
                "outer_fold",
                "selected_features",
                "C",
                "gamma",
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
        SEARCH_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        OVERALL_SUMMARY_PATH,
        SELECTED_FEATURES_PATH,
        CONFUSION_MATRIX_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()