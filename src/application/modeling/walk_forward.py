from __future__ import annotations

import logging

import pandas as pd

from application.modeling.contracts import (
    FoldData,
    WalkForwardAdapter,
    WalkForwardRunConfig,
    WalkForwardRunResult,
)
from application.training.splits import build_timestamp_splits

logger = logging.getLogger(__name__)


class WalkForwardRunner:
    """Model-agnostic walk-forward OOS runner.

    Adapters own dataset shaping, fold fitting, and prediction mechanics. This
    runner owns timestamp splits, purge, OOS prediction normalization, and the
    common fold loop.
    """

    def __init__(self, *, config: WalkForwardRunConfig, adapter: WalkForwardAdapter) -> None:
        self._config = config
        self._adapter = adapter

    def run(self, dataset: pd.DataFrame) -> WalkForwardRunResult:
        prepared = self._adapter.prepare_dataset(dataset)
        if prepared.frame.empty:
            raise RuntimeError("Walk-forward: prepared dataset is empty.")

        splits = build_timestamp_splits(
            unique_ts=prepared.resolved_unique_timestamps(),
            n_splits=self._config.n_splits,
            split_mode=self._config.split_mode,
            monthly_train_months=self._config.monthly_train_months,
            monthly_test_months=self._config.monthly_test_months,
            monthly_window_mode=self._config.monthly_window_mode,
        )
        if not splits:
            raise RuntimeError("Walk-forward: no valid timestamp splits.")

        prediction_frames: list[pd.DataFrame] = []
        fold_details: list[dict] = []
        fold_predictions = []

        for fold_idx, original_train_ts, test_ts in splits:
            train_ts = original_train_ts
            if self._config.purge_gap > 0 and len(train_ts) > self._config.purge_gap:
                train_ts = train_ts[: -self._config.purge_gap]

            train_df = prepared.frame.loc[prepared.frame["timestamp"].isin(set(train_ts))].copy()
            test_df = prepared.frame.loc[prepared.frame["timestamp"].isin(set(test_ts))].copy()
            prediction_source = prepared.prediction_frame if prepared.prediction_frame is not None else prepared.frame
            if len(test_ts):
                prediction_test_df = prediction_source.loc[
                    (prediction_source["timestamp"] >= test_ts[0])
                    & (prediction_source["timestamp"] <= test_ts[-1])
                ].copy()
            else:
                prediction_test_df = prediction_source.iloc[0:0].copy()
            fold = FoldData(
                fold_idx=int(fold_idx),
                original_train_timestamps=original_train_ts,
                train_timestamps=train_ts,
                test_timestamps=test_ts,
                train_frame=train_df,
                test_frame=test_df,
                feature_columns=list(prepared.feature_columns),
                prediction_test_frame=prediction_test_df,
            )
            if train_df.empty or test_df.empty or self._adapter.should_skip_fold(fold):
                continue

            fold_prediction = self._adapter.fit_predict_fold(fold)
            predictions = self._normalize_predictions(fold_prediction.predictions, fold_idx=int(fold_idx))
            prediction_frames.append(predictions)
            fold_details.append(dict(fold_prediction.fold_details))
            fold_predictions.append(fold_prediction)

        if not prediction_frames:
            raise RuntimeError("Walk-forward: all folds were skipped, no predictions were produced.")

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
        if "decision_time" not in predictions.columns:
            predictions["decision_time"] = predictions["timestamp"]
        predictions["decision_time"] = pd.to_datetime(predictions["decision_time"], errors="coerce")
        predictions = (
            predictions.dropna(subset=["timestamp", "decision_time"])
            .sort_values(["decision_time", "symbol"])
            .reset_index(drop=True)
        )
        return WalkForwardRunResult(
            predictions=predictions,
            fold_details=fold_details,
            feature_columns=list(prepared.feature_columns),
            fold_predictions=fold_predictions,
        )

    @staticmethod
    def _normalize_predictions(predictions: pd.DataFrame, *, fold_idx: int) -> pd.DataFrame:
        required = {"timestamp", "symbol", "p_short", "p_long"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"Walk-forward prediction frame is missing columns: {missing}")

        normalized = predictions.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
        if "decision_time" not in normalized.columns:
            normalized["decision_time"] = normalized["timestamp"]
        normalized["decision_time"] = pd.to_datetime(normalized["decision_time"], errors="coerce")
        normalized["symbol"] = normalized["symbol"].astype(str)
        normalized["p_short"] = normalized["p_short"].astype(float)
        normalized["p_long"] = normalized["p_long"].astype(float)
        if "fold" not in normalized.columns:
            normalized["fold"] = int(fold_idx)
        else:
            normalized["fold"] = normalized["fold"].astype(int)
        return normalized[["timestamp", "symbol", "decision_time", "p_short", "p_long", "fold"]]
