from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, confidence_threshold: float = 0.55) -> dict:
    p_long = y_proba[:, 1]
    confidence_thresholds = sorted({round(max(0.5, confidence_threshold), 2), 0.55, 0.60, 0.65, 0.70})
    probability_threshold_metrics = {}
    p_short = y_proba[:, 0]
    y_true_series = y_true
    n_rows = len(y_true)
    for threshold in confidence_thresholds:
        signal = np.full(n_rows, -1, dtype=int)
        signal[p_long >= threshold] = 1
        signal[p_short >= threshold] = 0
        mask = signal != -1
        selected = int(mask.sum())
        coverage = float(selected / n_rows) if n_rows else 0.0
        if selected == 0:
            probability_threshold_metrics[f"{threshold:.2f}"] = {
                "rows": 0,
                "coverage": coverage,
                "long_signals": int((signal == 1).sum()),
                "short_signals": int((signal == 0).sum()),
                "no_trade": int((signal == -1).sum()),
                "signal_accuracy": None,
                "signal_balanced_accuracy": None,
                "signal_f1_macro": None,
            }
            continue
        subset_y_true = y_true_series[mask]
        subset_y_pred = signal[mask]
        long_signals = int((signal == 1).sum())
        short_signals = int((signal == 0).sum())
        long_tp = int(((signal == 1) & (y_true_series == 1)).sum())
        short_tp = int(((signal == 0) & (y_true_series == 0)).sum())
        total_true_long = int((y_true_series == 1).sum())
        total_true_short = int((y_true_series == 0).sum())
        probability_threshold_metrics[f"{threshold:.2f}"] = {
            "rows": selected,
            "coverage": coverage,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "no_trade": int((signal == -1).sum()),
            "signal_accuracy": float(accuracy_score(subset_y_true, subset_y_pred)),
            "signal_balanced_accuracy": float(balanced_accuracy_score(subset_y_true, subset_y_pred)),
            "signal_f1_macro": float(f1_score(subset_y_true, subset_y_pred, average="macro")),
            "long_precision": float(long_tp / long_signals) if long_signals > 0 else None,
            "short_precision": float(short_tp / short_signals) if short_signals > 0 else None,
            "long_recall_all": float(long_tp / total_true_long) if total_true_long > 0 else 0.0,
            "short_recall_all": float(short_tp / total_true_short) if total_true_short > 0 else 0.0,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, p_long)),
        "pr_auc": float(average_precision_score(y_true, p_long)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["short", "long"],
            output_dict=True,
            zero_division=0,
        ),
        "probability_threshold_metrics": probability_threshold_metrics,
    }
