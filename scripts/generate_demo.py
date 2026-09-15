from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from breast_mc import run_study, synthetic_microcalcifications

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    frame = synthetic_microcalcifications()
    result = run_study(frame)
    train, validation, test = result.split.patient_sets
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].bar(
        ["Train", "Validation", "Test"],
        [len(train), len(validation), len(test)],
        color=["#4472C4", "#70AD47", "#ED7D31"],
    )
    axes[0].set_title("Patient-disjoint partitions")
    axes[0].set_ylabel("patients")
    sns.heatmap(
        result.patient_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=axes[1],
    )
    axes[1].set_title(
        f"Patient aggregation\nF1 = {result.patient_f1:.3f} (synthetic only)"
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.tight_layout()
    output = ROOT / "docs" / "figures" / "grouped_evaluation.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
