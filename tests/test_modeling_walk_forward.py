from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from application.modeling.contracts import FoldData, FoldPrediction, ModelingDataset, WalkForwardRunConfig
from application.modeling.adapters.factory import create_walk_forward_adapter
from application.modeling.adapters import lightgbm as lightgbm_adapter_module
from application.modeling.adapters.lightgbm import LightGbmWalkForwardAdapter
from application.modeling.walk_forward import WalkForwardRunner
from core.config.train_config import TrainConfig


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
        prediction_columns = ["timestamp", "symbol"]
        if "decision_time" in fold.test_frame.columns:
            prediction_columns.append("decision_time")
        predictions = fold.test_frame[prediction_columns].copy()
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
    assert list(result.predictions.columns) == ["timestamp", "symbol", "decision_time", "p_short", "p_long", "fold"]
    assert result.predictions["symbol"].tolist() == sorted(result.predictions["symbol"].tolist())
    assert result.predictions["p_long"].eq(0.75).all()
    assert result.fold_details[0]["train_timestamps"] == len(adapter.folds[0].train_timestamps)
    assert result.feature_columns == ["feature"]


def test_walk_forward_runner_preserves_explicit_decision_time() -> None:
    frame = make_frame()
    frame["decision_time"] = frame["timestamp"] + pd.Timedelta(hours=1)
    adapter = RecordingAdapter()
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

    result = runner.run(frame)

    assert "decision_time" in result.predictions.columns
    assert (result.predictions["decision_time"] > result.predictions["timestamp"]).all()


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


def test_create_walk_forward_adapter_returns_registered_lightgbm_adapter() -> None:
    adapter = create_walk_forward_adapter("lightgbm", train_cfg=TrainConfig.from_env())

    assert isinstance(adapter, LightGbmWalkForwardAdapter)


def test_lightgbm_walk_forward_predicts_all_test_rows_without_target_hindsight(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict(self, x):
            return np.ones(len(x), dtype=int)

        def predict_proba(self, x):
            proba = np.zeros((len(x), 2), dtype=float)
            proba[:, 0] = 0.25
            proba[:, 1] = 0.75
            return proba

    def fake_fit_model_with_internal_eval(**_kwargs):
        return FakeModel(), 10, {}

    monkeypatch.setattr(lightgbm_adapter_module, "fit_model_with_internal_eval", fake_fit_model_with_internal_eval)

    cfg = TrainConfig.from_env()
    cfg.use_symbol_feature = False
    cfg.enable_feature_clip = False
    cfg.regime_aware_weighting = False
    cfg.train_feature_subset = ""
    frame = make_frame(periods=12)
    frame["Target"] = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0]
    adapter = LightGbmWalkForwardAdapter(train_cfg=cfg)
    prepared = adapter.prepare_dataset(frame)
    prediction_frame = prepared.prediction_frame
    assert prediction_frame is not None
    neutral_timestamps = set(frame.loc[frame["Target"] == 0, "timestamp"])
    assert not prepared.frame["timestamp"].isin(neutral_timestamps).any()
    assert prediction_frame["Target"].eq(0).any()
    fold = FoldData(
        fold_idx=1,
        original_train_timestamps=prepared.unique_timestamps[:6],
        train_timestamps=prepared.unique_timestamps[:6],
        test_timestamps=prepared.unique_timestamps[6:],
        train_frame=prepared.frame.loc[prepared.frame["timestamp"].isin(prepared.unique_timestamps[:6])].copy(),
        test_frame=prepared.frame.loc[prepared.frame["timestamp"].isin(prepared.unique_timestamps[6:])].copy(),
        feature_columns=prepared.feature_columns,
        prediction_test_frame=prediction_frame.loc[
            (prediction_frame["timestamp"] >= prepared.unique_timestamps[6])
            & (prediction_frame["timestamp"] <= prepared.unique_timestamps[-1])
        ].copy(),
    )

    result = adapter.fit_predict_fold(fold)

    assert len(result.predictions) == len(fold.prediction_test_frame)
    assert result.fold_details["prediction_rows"] == len(fold.prediction_test_frame)
    assert result.fold_details["test_rows"] == len(fold.test_frame)
    assert result.predictions["timestamp"].isin(neutral_timestamps).any()
    assert len(result.y_true) == len(fold.test_frame)
