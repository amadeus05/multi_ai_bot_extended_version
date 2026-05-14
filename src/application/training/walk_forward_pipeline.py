from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from application.modeling.adapters.lightgbm import LightGbmWalkForwardAdapter
from application.modeling.contracts import WalkForwardRunConfig
from application.modeling.walk_forward import WalkForwardRunner
from core.config.train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    fold_details: list[dict]
    feature_columns: list[str]


class WalkForwardPipeline:
    """Backward-compatible LightGBM walk-forward facade over the common runner."""

    def __init__(self, *, train_cfg: TrainConfig) -> None:
        self.train_cfg = train_cfg

    def run(self, dataset: pd.DataFrame) -> WalkForwardResult:
        runner = WalkForwardRunner(
            config=WalkForwardRunConfig(
                n_splits=self.train_cfg.n_splits,
                split_mode=self.train_cfg.split_mode,
                monthly_train_months=self.train_cfg.monthly_train_months,
                monthly_test_months=self.train_cfg.monthly_test_months,
                monthly_window_mode=self.train_cfg.monthly_window_mode,
                purge_gap=self.train_cfg.purge_gap,
            ),
            adapter=LightGbmWalkForwardAdapter(train_cfg=self.train_cfg),
        )
        result = runner.run(dataset)
        return WalkForwardResult(
            predictions=result.predictions,
            fold_details=result.fold_details,
            feature_columns=result.feature_columns,
        )

    @staticmethod
    def save_artifacts(predictions: pd.DataFrame, fold_details: list[dict], train_cfg: TrainConfig) -> Path:
        out_dir = train_cfg.models_dir_path
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "walk_forward_oos_predictions.csv"
        predictions.to_csv(csv_path, index=False)

        ts_min = predictions["timestamp"].min()
        ts_max = predictions["timestamp"].max()
        summary = {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbols": list(train_cfg.symbols),
            "prediction_rows": int(len(predictions)),
            "prediction_period": {
                "start": str(ts_min),
                "end": str(ts_max),
            },
            "fold_details": fold_details,
        }
        json_path = out_dir / "walk_forward_oos_predictions_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Walk-forward artifacts: %s, %s", csv_path, json_path)
        return csv_path
