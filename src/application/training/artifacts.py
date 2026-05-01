from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from domain.ml.features import MasterFeatureBuilder
from domain.ml.features import config as feature_cfg
from domain.ml.features.models.feature_spec import serialize_feature_specs


def save_artifacts(
    models_dir: Path,
    model_name: str,
    model,
    metrics: dict,
    feature_columns: list[str],
    symbols: list[str],
    fold_importance: pd.DataFrame,
    clip_bounds: dict[str, dict[str, float]] | None = None,
) -> dict[str, str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.joblib"
    metrics_path = models_dir / f"{model_name}_metrics.json"
    features_path = models_dir / f"{model_name}_features.json"
    importance_path = models_dir / f"{model_name}_feature_importance.csv"
    fold_importance_path = models_dir / f"{model_name}_fold_feature_importance.csv"
    formulas_path = models_dir / f"{model_name}_feature_formulas.json"
    history_path = models_dir / f"{model_name}_train_history.json"
    clip_bounds_path = models_dir / f"{model_name}_clip_bounds.json"
    artifact_paths = [
        model_path,
        metrics_path,
        features_path,
        importance_path,
        fold_importance_path,
        formulas_path,
        history_path,
        clip_bounds_path,
    ]

    backup_existing_artifacts(artifact_paths, model_name, models_dir)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "symbols": symbols,
                "label_mapping": {"short": 0, "long": 1},
                "inverse_label_mapping": {"0": -1, "1": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(importance_path, index=False)
    if not fold_importance.empty:
        fold_importance.to_csv(fold_importance_path, index=False)

    builder = MasterFeatureBuilder()
    tracked = [column for column in feature_columns if column != "symbol"]
    specs = builder.collect_feature_specs(set(tracked))
    features_payload = serialize_feature_specs(specs, feature_cfg)
    formulas_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "features": features_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if clip_bounds:
        clip_bounds_path.write_text(json.dumps(clip_bounds, indent=2), encoding="utf-8")
    oos_metrics = metrics.get("oos_metrics", {})
    configured_threshold = float(metrics.get("confidence_threshold", 0.55))
    threshold_key = f"{max(0.5, configured_threshold):.2f}"
    threshold_metrics = (oos_metrics.get("probability_threshold_metrics") or {}).get(threshold_key, {})
    fold_stability = metrics.get("fold_stability", {})
    history_entry = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_name": model_name,
        "accuracy": oos_metrics.get("accuracy"),
        "balanced_accuracy": oos_metrics.get("balanced_accuracy"),
        "f1_macro": oos_metrics.get("f1_macro"),
        "roc_auc": oos_metrics.get("roc_auc"),
        "pr_auc": oos_metrics.get("pr_auc"),
        "mcc": oos_metrics.get("mcc"),
        "configured_threshold": configured_threshold,
        "signal_accuracy": threshold_metrics.get("signal_accuracy"),
        "signal_coverage": threshold_metrics.get("coverage"),
        "signal_rows": threshold_metrics.get("rows"),
        "fold_stability_pct": (
            float(fold_stability.get("accuracy_std")) * 100
            if fold_stability.get("accuracy_std") is not None
            else None
        ),
        "median_best_iteration": metrics.get("median_best_iteration"),
        "total_rows": metrics.get("total_rows"),
        "feature_count": metrics.get("feature_count"),
    }
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(history_entry)
    history = history[-200:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "features": str(features_path),
        "importance": str(importance_path),
        "fold_importance": str(fold_importance_path),
        "feature_formulas": str(formulas_path),
        "train_history": str(history_path),
        "clip_bounds": str(clip_bounds_path),
    }


def backup_existing_artifacts(paths: list[Path], model_name: str, models_dir: Path) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = models_dir / "backups" / f"{model_name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_paths:
        shutil.copy2(path, backup_dir / path.name)


def load_train_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def format_compact_metric_value(value, percent: bool = False, decimals: int = 4) -> str:
    if value is None:
        return "-"
    if percent:
        return f"{float(value):.{decimals}f}%"
    return f"{float(value):.{decimals}f}"


def build_current_run_summary_lines(history_entry: dict) -> list[str]:
    configured_threshold = float(history_entry.get("configured_threshold", 0.55))
    rows = [
        ("Accuracy", format_compact_metric_value(float(history_entry.get("accuracy", 0.0)) * 100, percent=True, decimals=2)),
        (
            f"Signal acc @{configured_threshold:.2f}",
            format_compact_metric_value(
                float(history_entry["signal_accuracy"]) * 100 if history_entry.get("signal_accuracy") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        (
            f"Coverage @{configured_threshold:.2f}",
            format_compact_metric_value(
                float(history_entry["signal_coverage"]) * 100 if history_entry.get("signal_coverage") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", format_compact_metric_value(history_entry.get("mcc"), decimals=3)),
        ("ROC AUC", format_compact_metric_value(history_entry.get("roc_auc"), decimals=3)),
        ("PR AUC", format_compact_metric_value(history_entry.get("pr_auc"), decimals=3)),
        ("Fold stability", format_compact_metric_value(history_entry.get("fold_stability_pct"), percent=True, decimals=2)),
    ]
    metric_width = max(len("Metric"), *(len(name) for name, _ in rows))
    value_width = max(len("Current"), *(len(value) for _, value in rows))
    border = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    lines = [border, f"| {'Metric'.ljust(metric_width)} | {'Current'.ljust(value_width)} |", border]
    for name, value in rows:
        lines.append(f"| {name.ljust(metric_width)} | {value.ljust(value_width)} |")
    lines.append(border)
    return lines


def build_recent_runs_table_lines(history: list[dict], limit: int = 10) -> list[str]:
    recent_entries = list(reversed(history[-limit:]))
    if not recent_entries:
        return ["No train history yet."]
    columns = [
        ("Run", lambda item: str(item.get("run_timestamp_utc", ""))[5:16]),
        ("Acc", lambda item: format_compact_metric_value(float(item.get("accuracy", 0.0)) * 100, percent=True, decimals=2)),
        (
            "Sig",
            lambda item: format_compact_metric_value(
                float(item["signal_accuracy"]) * 100 if item.get("signal_accuracy") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        (
            "Cov",
            lambda item: format_compact_metric_value(
                float(item["signal_coverage"]) * 100 if item.get("signal_coverage") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", lambda item: format_compact_metric_value(item.get("mcc"), decimals=3)),
        ("ROC", lambda item: format_compact_metric_value(item.get("roc_auc"), decimals=3)),
        ("PR", lambda item: format_compact_metric_value(item.get("pr_auc"), decimals=3)),
        ("Stab", lambda item: format_compact_metric_value(item.get("fold_stability_pct"), percent=True, decimals=2)),
        ("Rows", lambda item: str(item.get("total_rows", "-"))),
    ]
    rendered_rows = [[formatter(entry) for _, formatter in columns] for entry in recent_entries]
    widths = [max(len(header), *(len(row[idx]) for row in rendered_rows)) for idx, (header, _) in enumerate(columns)]

    def render_border():
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render_row(values):
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [render_border(), render_row([header for header, _ in columns]), render_border()]
    for row in rendered_rows:
        lines.append(render_row(row))
    lines.append(render_border())
    return lines
