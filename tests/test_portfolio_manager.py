import asyncio

import pytest

from core.types.domain_types import Trade
from core.types.enums import OrderSide, PositionSide
from domain.portfolio.portfolio_manager import PortfolioManager


def trade(
    *,
    side: OrderSide,
    amount: float,
    price: float,
    fee: float = 0.0,
    symbol: str = "BTC/USDT",
) -> Trade:
    return Trade(
        order_id=f"{side.value}-{amount}-{price}",
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        fee=fee,
    )


def test_opening_perp_position_charges_fee_but_does_not_spend_notional() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})

    portfolio.apply_execution(trade(side=OrderSide.BUY, amount=1.0, price=100.0, fee=0.4))

    position = portfolio.get_position("BTC/USDT")
    assert position is not None
    assert position.side == PositionSide.LONG
    assert position.amount == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(100.0)
    assert portfolio.cash["USDT"] == pytest.approx(999.6)


def test_closing_long_realizes_gross_pnl_and_net_closed_result_after_fees() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    portfolio.apply_execution(trade(side=OrderSide.BUY, amount=1.0, price=100.0, fee=0.4))

    portfolio.apply_execution(trade(side=OrderSide.SELL, amount=1.0, price=110.0, fee=0.4))

    assert portfolio.get_position("BTC/USDT") is None
    assert portfolio.realized_pnl == pytest.approx(10.0)
    assert portfolio.cash["USDT"] == pytest.approx(1009.2)
    assert portfolio.closed_trade_results[-1]["pnl_abs_gross"] == pytest.approx(10.0)
    assert portfolio.closed_trade_results[-1]["commission"] == pytest.approx(0.8)
    assert portfolio.closed_trade_results[-1]["pnl_abs"] == pytest.approx(9.2)


def test_partial_close_reduces_position_and_allocates_entry_fee_proportionally() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    portfolio.apply_execution(trade(side=OrderSide.BUY, amount=2.0, price=100.0, fee=2.0))

    portfolio.apply_execution(trade(side=OrderSide.SELL, amount=1.0, price=110.0, fee=1.0))

    position = portfolio.get_position("BTC/USDT")
    assert position is not None
    assert position.side == PositionSide.LONG
    assert position.amount == pytest.approx(1.0)
    assert portfolio.realized_pnl == pytest.approx(10.0)
    assert portfolio.cash["USDT"] == pytest.approx(1007.0)
    assert portfolio.closed_trade_results[-1]["commission"] == pytest.approx(2.0)
    assert portfolio.closed_trade_results[-1]["pnl_abs"] == pytest.approx(8.0)


def test_over_close_flips_position_and_allocates_exit_fee_between_close_and_new_entry() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    portfolio.apply_execution(trade(side=OrderSide.BUY, amount=1.0, price=100.0, fee=1.0))

    portfolio.apply_execution(trade(side=OrderSide.SELL, amount=2.0, price=110.0, fee=2.0))

    position = portfolio.get_position("BTC/USDT")
    assert position is not None
    assert position.side == PositionSide.SHORT
    assert position.amount == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(110.0)
    assert portfolio.cash["USDT"] == pytest.approx(1007.0)
    assert portfolio.closed_trade_results[-1]["commission"] == pytest.approx(2.0)
    assert portfolio.closed_trade_results[-1]["pnl_abs"] == pytest.approx(8.0)


def test_revalue_positions_returns_cash_plus_unrealized_pnl_for_linear_perps() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    portfolio.apply_execution(trade(side=OrderSide.BUY, amount=2.0, price=100.0))

    equity = asyncio.run(portfolio.revalue_positions({"BTC/USDT": 110.0}))

    assert portfolio.unrealized_pnl == pytest.approx(20.0)
    assert equity == pytest.approx(1020.0)


def test_revalue_positions_handles_short_unrealized_pnl_symmetrically() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    portfolio.apply_execution(trade(side=OrderSide.SELL, amount=2.0, price=100.0))

    equity = asyncio.run(portfolio.revalue_positions({"BTC/USDT": 90.0}))

    assert portfolio.unrealized_pnl == pytest.approx(20.0)
    assert equity == pytest.approx(1020.0)
