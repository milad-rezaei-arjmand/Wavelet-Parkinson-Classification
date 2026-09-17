from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

WPT_FEATURE_PATH = OUTPUTS_DIR / "wpt_features_db4_level4.parquet"
SWT_FEATURE_PATH = OUTPUTS_DIR / "swt_features_db4_level5.parquet"

RESULTS_DIR = OUTPUTS_DIR / "wpt_swt_representation_screening"
FIGURES_DIR = RESULTS_DIR / "figures"

FOLD_RESULTS_PATH = RESULTS_DIR / "representation_fold_results.csv"
OVERALL_RESULTS_PATH = RESULTS_DIR / "representation_overall_results.csv"
SUBJECT_PREDICTIONS_PATH = RESULTS_DIR / "representation_subject_predictions.csv"
SELECTED_FEATURES_PATH = RESULTS_DIR / "selected_feature_frequency.csv"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
SELECT_K = 300
SVM_C = 10.0
SVM_GAMMA = "scale"
EPSILON = 1e-12

LABEL_TO_ID = {"Healthy": 0, "Parkinson": 1}
ID_TO_LABEL = {0: "Healthy", 1: "Parkinson"}

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
# Loading and alignment
# ============================================================

def normalize_subject_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(3)
    )
    return dataframe


def load_representations() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, list[str]],
]:
    if not WPT_FEATURE_PATH.exists():
        raise FileNotFoundError(f"WPT file not found: {WPT_FEATURE_PATH}")
    if not SWT_FEATURE_PATH.exists():
        raise FileNotFoundError(f"SWT file not found: {SWT_FEATURE_PATH}")

    print("Loading WPT features...")
    wpt = normalize_subject_ids(pd.read_parquet(WPT_FEATURE_PATH))

    print("Loading SWT features...")
    swt = normalize_subject_ids(pd.read_parquet(SWT_FEATURE_PATH))

    for name, dataframe in [("WPT", wpt), ("SWT", swt)]:
        missing = set(META_COLUMNS) - set(dataframe.columns)
        if missing:
            raise ValueError(f"{name} metadata columns missing: {sorted(missing)}")

    wpt = wpt.sort_values(META_COLUMNS).reset_index(drop=True)
    swt = swt.sort_values(META_COLUMNS).reset_index(drop=True)

    if len(wpt) != 15975 or len(swt) != 15975:
        raise ValueError(
            f"Unexpected row counts: WPT={len(wpt)}, SWT={len(swt)}"
        )

    wpt_alignment = wpt[META_COLUMNS].astype(str)
    swt_alignment = swt[META_COLUMNS].astype(str)

    if not wpt_alignment.equals(swt_alignment):
        mismatch_count = int((wpt_alignment != swt_alignment).any(axis=1).sum())
        raise RuntimeError(
            f"WPT and SWT rows are not aligned. Mismatched rows: {mismatch_count}"
        )

    metadata = wpt[META_COLUMNS].copy()

    if metadata["subject_id"].nunique() != 355:
        raise ValueError("Expected exactly 355 subjects.")
    if not np.all(metadata.groupby("subject_id").size() == 45):
        raise ValueError("Not every subject has exactly 45 windows.")
    if not np.all(metadata.groupby("subject_id")["activity"].nunique() == 11):
        raise ValueError("Not every subject has exactly 11 activities.")

    unknown_labels = set(metadata["label"].unique()) - set(LABEL_TO_ID)
    if unknown_labels:
        raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    y = metadata["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    groups = metadata["subject_id"].to_numpy()

    wpt_columns = [column for column in wpt.columns if column not in META_COLUMNS]
    swt_columns = [column for column in swt.columns if column not in META_COLUMNS]

    if len(wpt_columns) != 1728:
        raise ValueError(f"Unexpected WPT feature count: {len(wpt_columns)}")
    if len(swt_columns) != 648:
        raise ValueError(f"Unexpected SWT feature count: {len(swt_columns)}")

    print("Converting WPT features to NumPy...")
    X_wpt = wpt[wpt_columns].to_numpy(dtype=np.float32, copy=True)

    print("Converting SWT features to NumPy...")
    X_swt = swt[swt_columns].to_numpy(dtype=np.float32, copy=True)

    if not np.all(np.isfinite(X_wpt)):
        raise ValueError("WPT matrix contains NaN or Inf.")
    if not np.all(np.isfinite(X_swt)):
        raise ValueError("SWT matrix contains NaN or Inf.")

    print("Creating WPT + SWT fusion matrix...")
    X_fusion = np.concatenate([X_wpt, X_swt], axis=1)

    if X_fusion.shape[1] != 2376:
        raise RuntimeError(f"Unexpected fusion feature count: {X_fusion.shape[1]}")

    representations = {
        "WPT": X_wpt,
        "SWT": X_swt,
        "WPT_SWT_Fusion": X_fusion,
    }

    feature_names = {
        "WPT": [f"WPT__{name}" for name in wpt_columns],
        "SWT": [f"SWT__{name}" for name in swt_columns],
        "WPT_SWT_Fusion": (
            [f"WPT__{name}" for name in wpt_columns]
            + [f"SWT__{name}" for name in swt_columns]
        ),
    }

    del wpt, swt, wpt_alignment, swt_alignment
    gc.collect()

    return metadata, y, groups, representations, feature_names


# ============================================================
# Model and scores
# ============================================================

def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=SELECT_K)),
            ("scale", StandardScaler()),
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


def get_decision_scores(fitted_model: Pipeline, X: np.ndarray) -> np.ndarray:
    scores = np.asarray(fitted_model.decision_function(X), dtype=float).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Decision scores contain NaN or Inf.")
    return scores


# ============================================================
# Subject aggregation
# ============================================================

def aggregate_subject_scores(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    temporary = metadata[["subject_id", "activity"]].copy()
    temporary["y_true"] = np.asarray(y_true, dtype=np.int8)
    temporary["score"] = np.asarray(scores, dtype=float)

    label_counts = temporary.groupby(["subject_id", "activity"])["y_true"].nunique()
    if not np.all(label_counts == 1):
        raise ValueError("Inconsistent labels inside subject/activity groups.")

    activity_scores = (
        temporary.groupby(["subject_id", "activity"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            activity_score=("score", "mean"),
            n_windows=("score", "size"),
        )
    )

    subject_scores = (
        activity_scores.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            score=("activity_score", "mean"),
            n_activities=("activity", "nunique"),
            n_windows=("n_windows", "sum"),
        )
    )

    if not np.all(subject_scores["n_activities"] == 11):
        raise ValueError("Not every subject has exactly 11 activities.")
    if not np.all(subject_scores["n_windows"] == 45):
        raise ValueError("Not every subject has exactly 45 windows.")

    return subject_scores


# ============================================================
# Metrics and threshold tuning
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
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
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def create_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(np.asarray(scores, dtype=float))

    if unique_scores.size == 0:
        raise ValueError("No scores available for threshold tuning.")

    if unique_scores.size == 1:
        return np.asarray(
            [
                unique_scores[0] - 1e-6,
                unique_scores[0],
                unique_scores[0] + 1e-6,
            ],
            dtype=float,
        )

    middle_points = (unique_scores[:-1] + unique_scores[1:]) / 2.0

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


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None

    for threshold in create_threshold_candidates(scores):
        y_pred = (scores >= threshold).astype(np.int8)
        metrics = calculate_metrics(y_true, y_pred, scores)

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

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Threshold tuning failed.")

    return best_threshold, best_metrics


# ============================================================
# Feature tracking and output helpers
# ============================================================

def get_selected_feature_names(
    fitted_model: Pipeline,
    original_feature_names: list[str],
) -> list[str]:
    variance_selector = fitted_model.named_steps["variance"]
    feature_selector = fitted_model.named_steps["select"]

    after_variance = np.asarray(original_feature_names)[
        variance_selector.get_support()
    ]

    return after_variance[feature_selector.get_support()].tolist()


def safe_file_name(text: str) -> str:
    return text.lower().replace(" ", "_").replace("+", "plus")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    representation: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Healthy", "Parkinson"],
    )
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(f"{representation} — subject-level OOF")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / f"{safe_file_name(representation)}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    representation: str,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Healthy", "Parkinson"],
        digits=4,
        zero_division=0,
    )

    output_path = (
        RESULTS_DIR
        / f"{safe_file_name(representation)}_classification_report.txt"
    )
    output_path.write_text(report, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metadata, y, groups, representations, feature_names = load_representations()

    print("\n" + "=" * 90)
    print("WPT / SWT / FUSION REPRESENTATION SCREENING")
    print("=" * 90)

    print(f"\nRows: {len(metadata)}")
    print(f"Subjects: {metadata['subject_id'].nunique()}")
    print(f"Outer folds: {N_OUTER_SPLITS}")
    print(f"Inner folds: {N_INNER_SPLITS}")
    print(f"Selected features: {SELECT_K}")

    for representation, X in representations.items():
        print(f"{representation} features: {X.shape[1]}")

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

    model_template = create_model()

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_counters = {
        representation: Counter()
        for representation in representations
    }

    for outer_fold, (outer_train_indices, outer_test_indices) in enumerate(
        outer_splits,
        start=1,
    ):
        train_subjects = set(groups[outer_train_indices])
        test_subjects = set(groups[outer_test_indices])

        if train_subjects.intersection(test_subjects):
            raise RuntimeError(
                f"Subject leakage detected in outer fold {outer_fold}."
            )

        print("\n" + "=" * 90)
        print(f"OUTER FOLD {outer_fold}/{N_OUTER_SPLITS}")
        print("=" * 90)
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

        for representation, X in representations.items():
            print(f"\nRepresentation: {representation}")

            inner_subject_frames: list[pd.DataFrame] = []

            for inner_train_relative, inner_validation_relative in inner_splits_relative:
                inner_train_indices = outer_train_indices[inner_train_relative]
                inner_validation_indices = outer_train_indices[
                    inner_validation_relative
                ]

                inner_train_subjects = set(groups[inner_train_indices])
                inner_validation_subjects = set(groups[inner_validation_indices])

                if inner_train_subjects.intersection(inner_validation_subjects):
                    raise RuntimeError("Subject leakage inside inner CV.")

                model = clone(model_template)
                model.fit(X[inner_train_indices], y[inner_train_indices])

                validation_scores = get_decision_scores(
                    fitted_model=model,
                    X=X[inner_validation_indices],
                )

                validation_subjects = aggregate_subject_scores(
                    metadata=metadata.iloc[inner_validation_indices],
                    y_true=y[inner_validation_indices],
                    scores=validation_scores,
                )

                inner_subject_frames.append(validation_subjects)

                del model
                gc.collect()

            inner_predictions = pd.concat(
                inner_subject_frames,
                ignore_index=True,
            )

            expected_subjects = set(groups[outer_train_indices])
            actual_subjects = set(inner_predictions["subject_id"])

            if actual_subjects != expected_subjects:
                raise RuntimeError("Inner OOF subject set is incomplete.")

            if inner_predictions["subject_id"].value_counts().ne(1).any():
                raise RuntimeError("Duplicated subjects in inner OOF predictions.")

            inner_y_true = inner_predictions["y_true"].to_numpy(dtype=np.int8)
            inner_scores = inner_predictions["score"].to_numpy(dtype=float)

            selected_threshold, inner_metrics = find_best_threshold(
                y_true=inner_y_true,
                scores=inner_scores,
            )

            inner_score_std = float(np.std(inner_scores))
            if inner_score_std <= EPSILON:
                inner_score_std = 1.0

            final_model = clone(model_template)
            final_model.fit(X[outer_train_indices], y[outer_train_indices])

            outer_window_scores = get_decision_scores(
                fitted_model=final_model,
                X=X[outer_test_indices],
            )

            outer_subjects = aggregate_subject_scores(
                metadata=metadata.iloc[outer_test_indices],
                y_true=y[outer_test_indices],
                scores=outer_window_scores,
            )

            outer_subjects["y_pred"] = (
                outer_subjects["score"] >= selected_threshold
            ).astype(np.int8)

            outer_subjects["normalized_margin"] = (
                (outer_subjects["score"] - selected_threshold)
                / inner_score_std
            )

            outer_metrics = calculate_metrics(
                y_true=outer_subjects["y_true"].to_numpy(dtype=np.int8),
                y_pred=outer_subjects["y_pred"].to_numpy(dtype=np.int8),
                scores=outer_subjects["score"].to_numpy(dtype=float),
            )

            selected_features = get_selected_feature_names(
                fitted_model=final_model,
                original_feature_names=feature_names[representation],
            )

            feature_counters[representation].update(selected_features)

            fold_rows.append(
                {
                    "representation": representation,
                    "outer_fold": outer_fold,
                    "train_subjects": len(train_subjects),
                    "test_subjects": len(test_subjects),
                    "raw_feature_count": X.shape[1],
                    "selected_feature_count": len(selected_features),
                    "selected_threshold": selected_threshold,
                    "inner_score_std": inner_score_std,
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

            outer_subjects["representation"] = representation
            outer_subjects["outer_fold"] = outer_fold
            outer_subjects["selected_threshold"] = selected_threshold
            outer_subjects["true_label"] = outer_subjects["y_true"].map(
                ID_TO_LABEL
            )
            outer_subjects["predicted_label"] = outer_subjects["y_pred"].map(
                ID_TO_LABEL
            )

            prediction_frames.append(outer_subjects)

            print(f"Selected threshold: {selected_threshold:.6f}")
            print(f"Inner Macro-F1:     {inner_metrics['macro_f1']:.4f}")
            print(f"Outer Accuracy:     {outer_metrics['accuracy']:.4f}")
            print(
                "Outer Balanced Acc: "
                f"{outer_metrics['balanced_accuracy']:.4f}"
            )
            print(f"Outer Macro-F1:     {outer_metrics['macro_f1']:.4f}")
            print(f"Healthy Recall:     {outer_metrics['healthy_recall']:.4f}")
            print(
                "Parkinson Recall:   "
                f"{outer_metrics['parkinson_recall']:.4f}"
            )
            print(f"ROC-AUC:            {outer_metrics['roc_auc']:.4f}")

            del final_model, inner_predictions, inner_subject_frames
            gc.collect()

    fold_table = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    expected_prediction_rows = 355 * len(representations)
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            "Unexpected subject prediction count. "
            f"Expected {expected_prediction_rows}, found {len(predictions)}."
        )

    overall_rows: list[dict[str, Any]] = []

    for representation in representations:
        current = predictions[
            predictions["representation"] == representation
        ].copy()

        if len(current) != 355 or current["subject_id"].nunique() != 355:
            raise RuntimeError(
                f"Invalid subject predictions for {representation}."
            )

        overall_metrics = calculate_metrics(
            y_true=current["y_true"].to_numpy(dtype=np.int8),
            y_pred=current["y_pred"].to_numpy(dtype=np.int8),
            scores=current["normalized_margin"].to_numpy(dtype=float),
        )

        current_folds = fold_table[
            fold_table["representation"] == representation
        ]

        overall_rows.append(
            {
                "representation": representation,
                "n_subjects": len(current),
                "raw_feature_count": representations[representation].shape[1],
                "selected_feature_count": SELECT_K,
                **overall_metrics,
                "fold_accuracy_mean": float(
                    current_folds["outer_accuracy"].mean()
                ),
                "fold_accuracy_std": float(
                    current_folds["outer_accuracy"].std()
                ),
                "fold_macro_f1_mean": float(
                    current_folds["outer_macro_f1"].mean()
                ),
                "fold_macro_f1_std": float(
                    current_folds["outer_macro_f1"].std()
                ),
                "fold_roc_auc_mean": float(
                    current_folds["outer_roc_auc"].mean()
                ),
                "fold_roc_auc_std": float(
                    current_folds["outer_roc_auc"].std()
                ),
            }
        )

        save_confusion_matrix(
            y_true=current["y_true"].to_numpy(dtype=np.int8),
            y_pred=current["y_pred"].to_numpy(dtype=np.int8),
            representation=representation,
        )

        save_classification_report(
            y_true=current["y_true"].to_numpy(dtype=np.int8),
            y_pred=current["y_pred"].to_numpy(dtype=np.int8),
            representation=representation,
        )

    overall_table = pd.DataFrame(overall_rows).sort_values(
        by=["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    )

    selected_feature_rows: list[dict[str, Any]] = []

    for representation, counter in feature_counters.items():
        for feature_name, selected_folds in counter.most_common():
            if feature_name.startswith("WPT__"):
                feature_source = "WPT"
            elif feature_name.startswith("SWT__"):
                feature_source = "SWT"
            else:
                feature_source = "Unknown"

            selected_feature_rows.append(
                {
                    "representation": representation,
                    "feature_source": feature_source,
                    "feature_name": feature_name,
                    "selected_folds": selected_folds,
                    "selection_rate": selected_folds / N_OUTER_SPLITS,
                }
            )

    selected_feature_table = pd.DataFrame(selected_feature_rows)

    fold_table.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    overall_table.to_csv(
        OVERALL_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        SUBJECT_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    selected_feature_table.to_csv(
        SELECTED_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 98)
    print("OVERALL WPT / SWT REPRESENTATION RESULTS")
    print("=" * 98)

    print(
        overall_table[
            [
                "representation",
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

    print("\nFusion selected-feature sources:")

    fusion_sources = (
        selected_feature_table[
            selected_feature_table["representation"] == "WPT_SWT_Fusion"
        ]
        .groupby("feature_source")["selected_folds"]
        .sum()
        .sort_values(ascending=False)
    )

    print(fusion_sources.to_string())

    print("\nCreated files:")
    for output_path in [
        FOLD_RESULTS_PATH,
        OVERALL_RESULTS_PATH,
        SUBJECT_PREDICTIONS_PATH,
        SELECTED_FEATURES_PATH,
        FIGURES_DIR,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
