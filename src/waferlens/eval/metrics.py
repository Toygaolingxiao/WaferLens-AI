"""L3 evaluation: one metric set for every route, nothing route-specific.

The comparison table in results/ is produced from these functions only, so a
number quoted anywhere in the project traces back to the same computation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_predictions(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
) -> dict:
    """Macro-F1 (primary), per-class recall, full report, confusion matrix."""
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class_recall = {label: report[label]["recall"] for label in labels}
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def format_comparison_row(route: str, result: dict, notes: str = "") -> str:
    """One Markdown row for the results comparison table."""
    recalls = result["per_class_recall"]
    worst = min(recalls, key=recalls.get)
    return (
        f"| {route} | {result['macro_f1']:.3f} | "
        f"{recalls[worst]:.3f} ({worst}) | {notes} |"
    )
