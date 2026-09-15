from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .data import PatientSplit, patient_grouped_split


@dataclass(frozen=True, slots=True)
class StudyResult:
    split: PatientSplit
    estimator: Pipeline
    threshold: float
    mc_f1: float
    mc_roc_auc: float
    patient_accuracy: float
    patient_f1: float
    patient_confusion_matrix: np.ndarray


def build_pipeline(feature_count: int, random_state: int = 42) -> Pipeline:
    """Build the microcalcification classifier and its learned preprocessing."""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive.")
    selected_count = min(50, feature_count)

    return Pipeline(
        [
            ("scale", MinMaxScaler()),
            (
                "select",
                SelectKBest(
                    partial(mutual_info_classif, random_state=random_state),
                    k=selected_count,
                ),
            ),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=["Patient ID", "Label"]), frame["Label"].astype(int)


def aggregate_patients(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    labels: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate binary microcalcification predictions into patient decisions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1].")
    rows = pd.DataFrame(
        {
            "Patient ID": patient_ids.to_numpy(),
            "mc_prediction": np.asarray(predictions, dtype=int),
            "Label": labels.to_numpy(dtype=int),
        }
    )
    grouped = rows.groupby("Patient ID").agg(
        malignant_ratio=("mc_prediction", "mean"),
        Label=("Label", "first"),
        microcalcifications=("mc_prediction", "size"),
    )
    grouped["prediction"] = (grouped["malignant_ratio"] > threshold).astype(int)
    return grouped.reset_index()


def select_threshold(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    labels: pd.Series,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose the malignant-ratio threshold using validation patients only."""
    candidates = np.arange(0.05, 1.0, 0.05) if thresholds is None else thresholds
    if len(candidates) == 0:
        raise ValueError("At least one threshold candidate is required.")
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        aggregated = aggregate_patients(patient_ids, predictions, labels, float(threshold))
        score = f1_score(aggregated["Label"], aggregated["prediction"], zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def run_study(frame: pd.DataFrame, random_state: int = 42) -> StudyResult:
    """Run patient-disjoint training, threshold selection, and held-out evaluation."""
    split = patient_grouped_split(frame, random_state=random_state)
    x_train, y_train = _xy(split.train)
    x_validation, y_validation = _xy(split.validation)
    x_test, y_test = _xy(split.test)
    estimator = build_pipeline(x_train.shape[1], random_state)
    estimator.fit(x_train, y_train)

    validation_predictions = estimator.predict(x_validation)
    threshold, _ = select_threshold(
        split.validation["Patient ID"], validation_predictions, y_validation
    )
    test_predictions = estimator.predict(x_test)
    test_probabilities = estimator.predict_proba(x_test)[:, 1]
    patients = aggregate_patients(
        split.test["Patient ID"], test_predictions, y_test, threshold
    )
    return StudyResult(
        split=split,
        estimator=estimator,
        threshold=threshold,
        mc_f1=float(f1_score(y_test, test_predictions, zero_division=0)),
        mc_roc_auc=float(roc_auc_score(y_test, test_probabilities)),
        patient_accuracy=float(accuracy_score(patients["Label"], patients["prediction"])),
        patient_f1=float(f1_score(patients["Label"], patients["prediction"], zero_division=0)),
        patient_confusion_matrix=confusion_matrix(
            patients["Label"], patients["prediction"], labels=[0, 1]
        ),
    )
