"""Classification metrics for the Task 2 corruption classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.corruptions import CLASS_NAMES


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # rows = true class, sums to 1
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=CLASS_NAMES)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": per_class,
        "confusion": cm,
        "confusion_normalized": cm_norm,
    }
