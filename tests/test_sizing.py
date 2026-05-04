import pytest

from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.risk.sizing import (
    barrier_nominal_under_margin_cap,
    compute_barrier_position_notional,
    estimated_margin_reserved_perps_linear,
)


def test_barrier_position_notional_is_limited_by_risk_then_leverage() -> None:
    notional, required_margin = compute_barrier_position_notional(
        balance_for_risk=1000.0,
        effective_risk=0.01,
        stop_pct=0.02,
        leverage=3.0,
    )

    assert notional == pytest.approx(500.0)
    assert required_margin == pytest.approx(500.0 / 3.0)


def test_barrier_position_notional_is_capped_by_max_leverage() -> None:
    notional, required_margin = compute_barrier_position_notional(
        balance_for_risk=1000.0,
        effective_risk=0.10,
        stop_pct=0.01,
        leverage=3.0,
    )

    assert notional == pytest.approx(3000.0)
    assert required_margin == pytest.approx(1000.0)


def test_barrier_nominal_respects_available_margin_after_existing_positions() -> None:
    notional, required_margin = barrier_nominal_under_margin_cap(
        total_cash_snapshot=1000.0,
        available_margin=100.0,
        effective_risk=0.10,
        stop_pct=0.01,
        leverage=3.0,
    )

    assert notional == pytest.approx(300.0)
    assert required_margin == pytest.approx(100.0)


def test_reserved_margin_uses_absolute_linear_notional_for_longs_and_shorts() -> None:
    positions = [
        Position("BTC/USDT", PositionSide.LONG, amount=2.0, entry_price=100.0),
        Position("ETH/USDT", PositionSide.SHORT, amount=3.0, entry_price=50.0),
    ]

    assert estimated_margin_reserved_perps_linear(positions, leverage=5.0) == pytest.approx(70.0)
