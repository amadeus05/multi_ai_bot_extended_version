import math

import pandas as pd

from domain.ml.features import config as cfg
from domain.ml.features.master_feature_builder import MasterFeatureBuilder


def _ohlcv_frame(timestamps: list[str], closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100.0,
        }
    )


def test_htf_features_are_lagged_before_asof_merge(monkeypatch) -> None:
    monkeypatch.setattr(
        cfg,
        "FEATURE_PROFILES",
        {"causal_test": ["return_4h_1"]},
    )
    monkeypatch.setattr(
        cfg,
        "FEATURE_BUILD_REQUEST",
        {"profile": "causal_test"},
    )

    base = _ohlcv_frame(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
            "2024-01-01 04:00:00",
            "2024-01-01 05:00:00",
            "2024-01-01 06:00:00",
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
        ],
        [100.0] * 10,
    )
    htf = _ohlcv_frame(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 04:00:00",
            "2024-01-01 08:00:00",
        ],
        [100.0, 110.0, 220.0],
    )

    result = MasterFeatureBuilder().build({"BTC/USDT": base}, {"BTC/USDT": htf})
    features = result.feature_map["BTC/USDT"].set_index("timestamp")

    assert pd.isna(features.loc[pd.Timestamp("2024-01-01 04:00:00"), "return_4h_1"])
    assert pd.isna(features.loc[pd.Timestamp("2024-01-01 07:00:00"), "return_4h_1"])
    assert features.loc[pd.Timestamp("2024-01-01 08:00:00"), "return_4h_1"] == math.log(110.0 / 100.0)
    assert features.loc[pd.Timestamp("2024-01-01 09:00:00"), "return_4h_1"] == math.log(110.0 / 100.0)


def test_asof_merge_uses_previous_htf_row_until_exact_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        cfg,
        "FEATURE_PROFILES",
        {"causal_test": ["return_4h_1"]},
    )
    monkeypatch.setattr(
        cfg,
        "FEATURE_BUILD_REQUEST",
        {"profile": "causal_test"},
    )

    base = _ohlcv_frame(
        [
            "2024-01-01 03:59:59",
            "2024-01-01 04:00:00",
            "2024-01-01 04:00:01",
        ],
        [100.0, 100.0, 100.0],
    )
    htf_features = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 04:00:00",
                    "2024-01-01 08:00:00",
                ]
            ),
            "return_4h_1": [1.0, 2.0, 3.0],
        }
    )

    builder = MasterFeatureBuilder()
    merged = builder._merge_main_and_htf(
        base_feature_map={"BTC/USDT": base},
        htf_feature_map={"BTC/USDT": htf_features},
        requested_features={"return_4h_1"},
    )["BTC/USDT"].set_index("timestamp")

    assert merged.loc[pd.Timestamp("2024-01-01 03:59:59"), "return_4h_1"] == 1.0
    assert merged.loc[pd.Timestamp("2024-01-01 04:00:00"), "return_4h_1"] == 2.0
    assert merged.loc[pd.Timestamp("2024-01-01 04:00:01"), "return_4h_1"] == 2.0
