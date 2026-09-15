from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class PatientSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def patient_sets(self) -> tuple[set, set, set]:
        return tuple(
            set(partition["Patient ID"].unique())
            for partition in (self.train, self.validation, self.test)
        )


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the patient identifier, binary labels, and numeric feature columns."""
    required = {"Patient ID", "Label"}
    if not required.issubset(frame.columns):
        raise ValueError("The frame must contain 'Patient ID' and 'Label' columns.")
    if frame["Patient ID"].isna().any():
        raise ValueError("Patient ID cannot contain missing values.")
    labels = set(frame["Label"].dropna().unique())
    if not labels or not labels.issubset({0, 1}):
        raise ValueError("Label must contain only binary values 0 and 1.")
    label_counts = frame.groupby("Patient ID")["Label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Every patient must have one consistent diagnostic label.")
    features = frame.drop(columns=["Patient ID", "Label"])
    if features.empty or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("At least one numeric feature column is required.")
    return frame.copy()


def patient_grouped_split(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> PatientSplit:
    """Split patient IDs before expanding their microcalcification rows."""
    checked = validate_frame(frame)
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")
    patients = checked.groupby("Patient ID", as_index=False)["Label"].first()
    train_validation, test = train_test_split(
        patients,
        test_size=test_size,
        stratify=patients["Label"],
        random_state=random_state,
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        stratify=train_validation["Label"],
        random_state=random_state,
    )

    def rows(ids: pd.Series) -> pd.DataFrame:
        return checked[checked["Patient ID"].isin(ids)].reset_index(drop=True)

    result = PatientSplit(
        rows(train["Patient ID"]),
        rows(validation["Patient ID"]),
        rows(test["Patient ID"]),
    )
    train_ids, validation_ids, test_ids = result.patient_sets
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Patient overlap detected after grouped splitting.")
    return result
