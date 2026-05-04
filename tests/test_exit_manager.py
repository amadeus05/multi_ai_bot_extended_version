import pytest

from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.execution.exit_manager import ExitManager


def test_long_stop_loss_exits_at_stop_price_with_sell_slippage() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=105.0,
        next_low=89.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "SL"
    assert exit_price == pytest.approx(90.0 * 0.999)


def test_long_stop_loss_gap_uses_next_open_when_open_is_beyond_stop() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=85.0,
        next_high=95.0,
        next_low=84.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "SL"
    assert exit_price == pytest.approx(85.0 * 0.999)


def test_long_take_profit_exits_at_take_price_with_sell_slippage() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=121.0,
        next_low=99.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "TP"
    assert exit_price == pytest.approx(120.0 * 0.999)


def test_short_stop_loss_exits_at_stop_price_with_buy_slippage() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.SHORT, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=111.0,
        next_low=95.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "SL"
    assert exit_price == pytest.approx(110.0 * 1.001)


def test_short_take_profit_exits_at_take_price_with_buy_slippage() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.SHORT, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=101.0,
        next_low=79.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "TP"
    assert exit_price == pytest.approx(80.0 * 1.001)


def test_stop_loss_wins_when_single_bar_touches_stop_and_take() -> None:
    manager = ExitManager(slippage=0.0)
    position = Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=125.0,
        next_low=85.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert reason == "SL"
    assert exit_price == pytest.approx(90.0)


def test_no_exit_when_bar_stays_inside_barriers() -> None:
    manager = ExitManager(slippage=0.001)
    position = Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)

    exit_price, reason = manager.check_causal_exit(
        position=position,
        next_open=100.0,
        next_high=119.0,
        next_low=91.0,
        stop_pct=0.10,
        take_pct=0.20,
    )

    assert exit_price is None
    assert reason is None
