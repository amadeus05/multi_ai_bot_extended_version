import asyncio

import numpy as np
import pandas as pd
import pytest

from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.execution.exit_manager import ExitManager
from domain.ml.labeling.barrier_policy import BarrierPolicy
from domain.ml.labeling.barrier_target_labeling import attach_barrier_columns, resolve_trade_exit
from domain.ml.labeling.models import AdaptiveHorizonConfig, BarrierConfig, LabelingConfig
from infrastructure.data_providers.historical_replay_provider import HistoricalMultiSymbolReplayProvider


def _labeling_cfg() -> LabelingConfig:
    return LabelingConfig(
        adaptive_horizon=AdaptiveHorizonConfig(
            enabled=True,
            base_horizon=12,
            min_horizon=8,
            max_horizon=20,
            vol_low=0.005,
            vol_high=0.025,
        ),
        barrier=BarrierConfig(
            use_dynamic_barriers=True,
            atr_multiplier=1.25,
            rvol_multiplier=0.75,
            tp_to_sl_ratio=2.0,
            min_pct=0.0075,
            max_pct=0.06,
            slippage=0.0003,
        ),
        realized_vol_column="realized_vol_1h",
    )


def _raw_feature_frame(rows: int = 40) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100.0 + idx * 0.35
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "open": close - 0.1,
            "high": close + 1.2,
            "low": close - 1.1,
            "close": close,
            "volume": 1000.0 + idx,
            "symbol": "BTC/USDT",
            "realized_vol_1h": 0.01 + idx * 0.0001,
        }
    )


def test_live_style_barrier_policy_matches_dataset_barrier_columns() -> None:
    cfg = _labeling_cfg()
    raw = _raw_feature_frame()

    dataset_frame = attach_barrier_columns(raw, cfg)
    live_stop, live_take = BarrierPolicy(cfg).barriers_for_last_row(raw)

    assert live_stop == pytest.approx(dataset_frame["barrier_stop_pct"].iloc[-1])
    assert live_take == pytest.approx(dataset_frame["barrier_take_pct"].iloc[-1])


def test_precomputed_backtest_barriers_are_used_without_recalculation() -> None:
    cfg = _labeling_cfg()
    precomputed = _raw_feature_frame()
    precomputed["barrier_stop_pct"] = np.linspace(0.011, 0.019, len(precomputed))
    precomputed["barrier_take_pct"] = precomputed["barrier_stop_pct"] * 3.0

    stop, take = BarrierPolicy(cfg).barriers_for_last_row(precomputed)

    assert stop == pytest.approx(precomputed["barrier_stop_pct"].iloc[-1])
    assert take == pytest.approx(precomputed["barrier_take_pct"].iloc[-1])
    recomputed = attach_barrier_columns(precomputed.drop(columns=["barrier_stop_pct", "barrier_take_pct"]), cfg)
    assert stop != pytest.approx(recomputed["barrier_stop_pct"].iloc[-1])


def test_historical_backtest_warmup_preserves_precomputed_barrier_columns() -> None:
    frame = _raw_feature_frame()
    frame["barrier_stop_pct"] = np.linspace(0.01, 0.02, len(frame))
    frame["barrier_take_pct"] = frame["barrier_stop_pct"] * 2.0
    provider = HistoricalMultiSymbolReplayProvider({"BTC/USDT": frame})

    warmup = pd.DataFrame()
    for _ticks in provider.iter_replay_batches():
        warmup = asyncio.run(provider.warmup("BTC/USDT", 10))
        break

    assert {"barrier_stop_pct", "barrier_take_pct"}.issubset(warmup.columns)
    assert warmup["barrier_stop_pct"].iloc[-1] == pytest.approx(frame["barrier_stop_pct"].iloc[0])


@pytest.mark.parametrize(
    ("position", "direction", "next_open", "next_high", "next_low"),
    [
        (Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0), 1, 100.0, 103.0, 98.0),
        (Position("BTC/USDT", PositionSide.SHORT, amount=1.0, entry_price=100.0), -1, 100.0, 102.0, 97.0),
    ],
)
def test_exit_manager_matches_canonical_resolve_trade_exit(position, direction, next_open, next_high, next_low) -> None:
    manager = ExitManager(slippage=0.0003)

    actual = manager.check_causal_exit(
        position=position,
        next_open=next_open,
        next_high=next_high,
        next_low=next_low,
        stop_pct=0.01,
        take_pct=0.02,
    )
    expected = resolve_trade_exit(
        direction=direction,
        entry_price=position.entry_price,
        next_open=next_open,
        next_high=next_high,
        next_low=next_low,
        stop_pct=0.01,
        take_pct=0.02,
        slippage=0.0003,
    )

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == expected[1]
