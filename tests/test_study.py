import numpy as np
import pandas as pd
import pytest

from breast_mc import (
    aggregate_patients,
    run_study,
    select_threshold,
    synthetic_microcalcifications,
)


def test_patient_aggregation() -> None:
    aggregated = aggregate_patients(
        pd.Series(["A", "A", "B", "B"]),
        np.array([1, 1, 0, 1]),
        pd.Series([1, 1, 0, 0]),
        threshold=0.75,
    )
    assert aggregated["prediction"].tolist() == [1, 0]
    assert aggregated["malignant_ratio"].tolist() == [1.0, 0.5]


def test_threshold_selection_uses_best_candidate() -> None:
    threshold, score = select_threshold(
        pd.Series(["A", "A", "B", "B"]),
        np.array([1, 0, 0, 0]),
        pd.Series([1, 1, 0, 0]),
        thresholds=np.array([0.25, 0.75]),
    )
    assert threshold == 0.25
    assert score == 1.0


def test_study_is_deterministic_and_grouped() -> None:
    frame = synthetic_microcalcifications(patient_count=60, feature_count=20, seed=7)
    first = run_study(frame, random_state=4)
    second = run_study(frame, random_state=4)
    assert first.threshold == second.threshold
    assert first.patient_f1 == second.patient_f1
    assert np.array_equal(first.patient_confusion_matrix, second.patient_confusion_matrix)
    assert first.mc_roc_auc > 0.65


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_patients(pd.Series(["A"]), np.array([0]), pd.Series([0]), 1.5)
