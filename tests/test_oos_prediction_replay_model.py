from __future__ import annotations

import pandas as pd

from application.inference.walk_forward_model import OosPredictionReplayModel, WalkForwardPredictionModel


def test_oos_prediction_replay_model_reads_probability_lookup() -> None:
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    model = OosPredictionReplayModel(
        lookup={(ts, "BTC/USDT"): (0.2, 0.8)},
        feature_columns=["close"],
    )

    prediction = model.predict(
        pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01T00:00:00"),
                    "symbol": "BTC/USDT",
                    "close": 100.0,
                }
            ]
        )
    )

    assert prediction["p_short"] == 0.2
    assert prediction["p_long"] == 0.8
    assert prediction["score"] == 0.6000000000000001


def test_walk_forward_prediction_model_alias_is_preserved() -> None:
    assert WalkForwardPredictionModel is OosPredictionReplayModel
