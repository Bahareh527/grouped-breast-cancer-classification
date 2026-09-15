from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .study import run_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run patient-grouped microcalcification classification."
    )
    parser.add_argument("spreadsheet", type=Path)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    frame = pd.read_excel(args.spreadsheet)
    result = run_study(frame, random_state=args.random_state)
    train, validation, test = result.split.patient_sets
    print(
        json.dumps(
            {
                "patients": {
                    "train": len(train),
                    "validation": len(validation),
                    "test": len(test),
                },
                "threshold": result.threshold,
                "microcalcification_f1": result.mc_f1,
                "microcalcification_roc_auc": result.mc_roc_auc,
                "patient_accuracy": result.patient_accuracy,
                "patient_f1": result.patient_f1,
                "patient_confusion_matrix": result.patient_confusion_matrix.tolist(),
            },
            indent=2,
        )
    )

