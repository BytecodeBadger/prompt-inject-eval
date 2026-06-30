from __future__ import annotations

from collections import defaultdict
from typing import Any
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0

    p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
    auprc = float(auc(r_curve, p_curve))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": auprc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "adversarial_block_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "benign_false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def per_category_block_rate(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    blocked: dict[str, int] = defaultdict(int)

    for row in rows:
        category = row.get("category", "unknown")
        label = int(row.get("label", 0))
        pred = int(row.get("predicted_label", 0))
        if label == 1:
            totals[category] += 1
            if pred == 1:
                blocked[category] += 1

    output: dict[str, float] = {}
    for category, total in totals.items():
        output[category] = (blocked[category] / total) if total else 0.0
    return output
