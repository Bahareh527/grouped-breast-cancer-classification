from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_microcalcifications(
    patient_count: int = 120,
    feature_count: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Create hierarchical synthetic data for software verification only."""
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    patient_labels = np.resize([0, 1], patient_count)
    rng.shuffle(patient_labels)
    for patient_index, label in enumerate(patient_labels):
        mc_count = int(rng.integers(8, 21))
        patient_effect = rng.normal(0, 0.35, feature_count)
        signal = np.zeros(feature_count)
        signal[: min(12, feature_count)] = label * 0.75
        features = rng.normal(size=(mc_count, feature_count)) + patient_effect + signal
        frame = pd.DataFrame(
            features,
            columns=[f"feature_{index:03d}" for index in range(feature_count)],
        )
        frame.insert(0, "Patient ID", f"P{patient_index:03d}")
        frame["Label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
