from __future__ import annotations

import numpy as np
import pandas as pd

from application.modeling.contracts import FoldData, FoldPrediction, ModelingDataset, WalkForwardRunConfig
from application.modeling.walk_forward import WalkForwardRunner


class RecordingAdapter:
    model_id = "recording"

    def __init__(self) -> None:
        self.folds: list[FoldData] = []

    def prepare_dataset(self, dataset: pd.DataFrame) -> ModelingDataset:
        return ModelingDataset(
            frame=dataset.copy(),
            feature_columns=["feature"],
            unique_timestamps=np.sort(dataset["timestamp"].unique()),
        )

    def should_skip_fold(self, fold: FoldData) -> bool:
        return False

    def fit_predict_fold(self, fold: FoldData) -> FoldPrediction:
        self.folds.append(fold)
        predictions = fold.test_frame[["timestamp", "symbol"]].copy()
        predictions["p_short"] = 0.25
        predictions["p_long"] = 0.75
        return FoldPrediction(
            predictions=predictions,
            fold_details={
                "fold": fold.fold_idx,
                "train_rows": len(fold.train_frame),
                "test_rows": len(fold.test_frame),
                "train_timestamps": len(fold.train_timestamps),
            },
        )


class SkipFirstFoldAdapter(RecordingAdapter):
    def should_skip_fold(self, fold: FoldData) -> bool:
        return fold.fold_idx == 1


def make_frame(periods: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * periods,
            "feature": np.arange(periods, dtype=float),
            "Target": [1, -1] * (periods // 2),
        }
    )


def test_walk_forward_runner_applies_purge_and_normalizes_predictions() -> None:
    adapter = RecordingAdapter()
    runner = WalkForwardRunner(
        config=WalkForwardRunConfig(
            n_splits=2,
            split_mode="tscv",
            monthly_train_months=1,
            monthly_test_months=1,
            monthly_window_mode="expanding",
            purge_gap=1,
        ),
        adapter=adapter,
    )

    result = runner.run(make_frame())

    assert len(adapter.folds) == 2
    assert len(adapter.folds[0].train_timestamps) == len(adapter.folds[0].original_train_timestamps) - 1
    assert list(result.predictions.columns) == ["timestamp", "symbol", "p_short", "p_long", "fold"]
    assert result.predictions["symbol"].tolist() == sorted(result.predictions["symbol"].tolist())
    assert result.predictions["p_long"].eq(0.75).all()
    assert result.fold_details[0]["train_timestamps"] == len(adapter.folds[0].train_timestamps)
    assert result.feature_columns == ["feature"]


def test_walk_forward_runner_skips_adapter_rejected_folds() -> None:
    adapter = SkipFirstFoldAdapter()
    runner = WalkForwardRunner(
        config=WalkForwardRunConfig(
            n_splits=2,
            split_mode="tscv",
            monthly_train_months=1,
            monthly_test_months=1,
            monthly_window_mode="expanding",
            purge_gap=0,
        ),
        adapter=adapter,
    )

    result = runner.run(make_frame())

    assert [details["fold"] for details in result.fold_details] == [2]
    assert result.predictions["fold"].unique().tolist() == [2]
