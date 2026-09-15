import numpy as np
import pandas as pd
import pytest

from breast_mc import patient_grouped_split, validate_frame


def small_frame(patient_count: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for patient in range(patient_count):
        label = patient % 2
        for _ in range(4):
            rows.append(
                {
                    "Patient ID": f"P{patient:03d}",
                    "f1": rng.normal(label, 1),
                    "f2": rng.normal(label, 1),
                    "Label": label,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_patient_overlap() -> None:
    split = patient_grouped_split(small_frame())
    train, validation, test = split.patient_sets
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == 40


def test_grouped_split_is_deterministic() -> None:
    first = patient_grouped_split(small_frame(), random_state=8)
    second = patient_grouped_split(small_frame(), random_state=8)
    assert first.patient_sets == second.patient_sets


def test_inconsistent_patient_label_is_rejected() -> None:
    frame = small_frame()
    frame.loc[1, "Label"] = 1
    with pytest.raises(ValueError, match="consistent"):
        validate_frame(frame)

