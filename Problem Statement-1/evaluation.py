"""Evaluation utilities: F1, recall, precision, confusion matrix."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate(y_true, y_pred, label: str = "") -> dict:
    """Print and return F1, recall, precision."""
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}F1={f1:.4f}  Recall={recall:.4f}  Precision={precision:.4f}")
    print(f"{prefix}Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    return {"f1": f1, "recall": recall, "precision": precision}


def evaluate_by_outcome(y_true, y_pred, outcomes) -> None:
    """Break down performance by call outcome."""
    df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "outcome": outcomes,
    })
    print("\nPer-outcome breakdown:")
    print(f"  {'outcome':<15s} {'total':>5s} {'pos':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    for outcome in sorted(df["outcome"].unique()):
        sub = df[df["outcome"] == outcome]
        tp = ((sub["true"] == True) & (sub["pred"] == True)).sum()
        fp = ((sub["true"] == False) & (sub["pred"] == True)).sum()
        fn = ((sub["true"] == True) & (sub["pred"] == False)).sum()
        pos = (sub["true"] == True).sum()
        print(f"  {outcome:<15s} {len(sub):5d} {pos:4d} {tp:4d} {fp:4d} {fn:4d}")
