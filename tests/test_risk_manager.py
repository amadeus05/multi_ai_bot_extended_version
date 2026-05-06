import pandas as pd
import pytest

from core.types.domain_types import Order, Position
from core.types.enums import OrderSide, PositionSide
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.adaptive_risk_manager import AdaptiveRiskManager
from domain.risk.models.risk_profile import RiskProfile


def make_order(symbol: str = "BTC/USDT", side: OrderSide = OrderSide.BUY) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        amount=0.0,
        price=100.0,
        meta={"barrier_stop_pct": 0.02},
    )


def make_profile(**overrides) -> RiskProfile:
    values = {
        "leverage": 3.0,
        "risk_per_trade": 0.01,
        "max_new_positions_per_bar": 10,
        "max_open_positions": 10,
        "sl_cooldown_bars": 2,
        "max_sl_per_day": 10,
        "min_position_notional": 10.0,
    }
    values.update(overrides)
    return RiskProfile(**values)


def test_risk_manager_sizes_order_from_barrier_stop_and_cash_snapshot() -> None:
    manager = AdaptiveRiskManager(make_profile())
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    order = make_order()

    sized = manager.check(order, portfolio, exchange=object())

    assert sized is order
    assert sized.amount == pytest.approx(5.0)


def test_risk_manager_rejects_second_new_position_in_same_bar_when_limit_is_one() -> None:
    manager = AdaptiveRiskManager(make_profile(max_new_positions_per_bar=1))
    manager.set_current_bar(pd.Timestamp("2024-01-01T00:00:00"))
    portfolio = PortfolioManager(cash={"USDT": 1000.0})

    first = manager.check(make_order("BTC/USDT"), portfolio, exchange=object())
    second = manager.check(make_order("ETH/USDT"), portfolio, exchange=object())

    assert first is not None
    assert second is None


def test_risk_manager_resets_new_position_limit_on_next_bar() -> None:
    manager = AdaptiveRiskManager(make_profile(max_new_positions_per_bar=1))
    portfolio = PortfolioManager(cash={"USDT": 1000.0})

    manager.set_current_bar(pd.Timestamp("2024-01-01T00:00:00"))
    assert manager.check(make_order("BTC/USDT"), portfolio, exchange=object()) is not None

    manager.set_current_bar(pd.Timestamp("2024-01-01T01:00:00"))
    assert manager.check(make_order("ETH/USDT"), portfolio, exchange=object()) is not None


def test_risk_manager_rejects_new_symbol_when_open_position_cap_is_reached() -> None:
    manager = AdaptiveRiskManager(make_profile(max_open_positions=1))
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)],
    )

    assert manager.check(make_order("ETH/USDT"), portfolio, exchange=object()) is None


def test_risk_manager_allows_reducing_existing_position_even_when_open_cap_is_reached() -> None:
    manager = AdaptiveRiskManager(make_profile(max_open_positions=1))
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[Position("BTC/USDT", PositionSide.LONG, amount=1.0, entry_price=100.0)],
    )
    reduce_order = Order("BTC/USDT", OrderSide.SELL, amount=10.0, price=100.0)

    checked = manager.check(reduce_order, portfolio, exchange=object())

    assert checked is reduce_order
    assert checked.amount == pytest.approx(1.0)


def test_risk_manager_blocks_symbol_until_sl_cooldown_expires() -> None:
    manager = AdaptiveRiskManager(make_profile(sl_cooldown_bars=2, max_sl_per_day=10))
    portfolio = PortfolioManager(cash={"USDT": 1000.0})

    manager.set_current_bar(pd.Timestamp("2024-01-01T00:00:00"))
    manager.register_trade_result(
        "BTC/USDT",
        pnl=-10.0,
        bar_ts=pd.Timestamp("2024-01-01T00:00:00"),
        stop_loss_hit=True,
    )

    assert manager.check(make_order("BTC/USDT"), portfolio, exchange=object()) is None
    manager.set_current_bar(pd.Timestamp("2024-01-01T01:00:00"))
    assert manager.check(make_order("BTC/USDT"), portfolio, exchange=object()) is None
    manager.set_current_bar(pd.Timestamp("2024-01-01T02:00:00"))
    assert manager.check(make_order("BTC/USDT"), portfolio, exchange=object()) is not None


def test_risk_manager_blocks_entries_after_daily_stop_loss_limit() -> None:
    manager = AdaptiveRiskManager(make_profile(max_sl_per_day=1))
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    ts = pd.Timestamp("2024-01-01T00:00:00")

    manager.set_current_bar(ts)
    manager.register_trade_result("BTC/USDT", pnl=-10.0, bar_ts=ts, stop_loss_hit=True)

    assert manager.check(make_order("ETH/USDT"), portfolio, exchange=object()) is None


def test_risk_manager_treats_zero_daily_stop_loss_limit_as_disabled() -> None:
    manager = AdaptiveRiskManager(make_profile(max_sl_per_day=0))
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    ts = pd.Timestamp("2024-01-01T00:00:00")

    manager.set_current_bar(ts)
    manager.register_trade_result("BTC/USDT", pnl=-10.0, bar_ts=ts, stop_loss_hit=True)

    assert manager.check(make_order("ETH/USDT"), portfolio, exchange=object()) is not None


def test_risk_manager_ignores_zero_reduced_risk_after_loss_streak() -> None:
    manager = AdaptiveRiskManager(
        make_profile(
            risk_per_trade=0.01,
            reduce_risk_after_consecutive_losses=1,
            reduced_risk_per_trade=0.0,
        )
    )
    portfolio = PortfolioManager(cash={"USDT": 1000.0})

    manager.register_trade_result("BTC/USDT", pnl=-10.0, bar_ts=pd.Timestamp("2024-01-01T00:00:00"))
    checked = manager.check(make_order("ETH/USDT"), portfolio, exchange=object())

    assert checked is not None
    assert checked.amount == pytest.approx(5.0)


def test_risk_manager_logs_risk_reduction_and_restore(capsys) -> None:
    manager = AdaptiveRiskManager(
        make_profile(
            risk_per_trade=0.01,
            reduce_risk_after_consecutive_losses=2,
            reduced_risk_per_trade=0.005,
        )
    )

    manager.register_trade_result("BTC/USDT", pnl=-1.0, bar_ts=pd.Timestamp("2024-10-07T17:00:00"))
    assert capsys.readouterr().out == ""

    manager.register_trade_result("BTC/USDT", pnl=-1.0, bar_ts=pd.Timestamp("2024-10-07T18:00:00"))
    reduced = capsys.readouterr().out
    assert "[2024-10-07 18:00:00] ⚠️ Loss streak 2: risk per trade reduced to 0.50%" in reduced

    manager.register_trade_result("BTC/USDT", pnl=-1.0, bar_ts=pd.Timestamp("2024-10-08T18:00:00"))
    assert capsys.readouterr().out == ""

    manager.register_trade_result("BTC/USDT", pnl=1.0, bar_ts=pd.Timestamp("2024-10-10T18:00:00"))
    restored = capsys.readouterr().out
    assert "[2024-10-10 18:00:00] ℹ️ Loss streak reset: risk per trade restored to 1.00%" in restored
