import asyncio

import pandas as pd
import pytest

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


def test_market_fill_price_applies_side_dependent_slippage() -> None:
    exchange = SimulatedExchange(slippage=0.01, execution_price_source=None)

    assert exchange.market_fill_price(100.0, OrderSide.BUY) == pytest.approx(101.0)
    assert exchange.market_fill_price(100.0, OrderSide.SELL) == pytest.approx(99.0)


def test_order_status_applies_slippage_and_taker_fee_to_regular_market_order() -> None:
    exchange = SimulatedExchange(commission=0.001, slippage=0.01, execution_price_source=None)
    order = Order("BTC/USDT", OrderSide.BUY, amount=2.0, price=100.0)

    order_id = asyncio.run(exchange.place_order(order))
    status = asyncio.run(exchange.get_order_status(order_id))

    assert status["filled"] == pytest.approx(2.0)
    assert status["avg_price"] == pytest.approx(101.0)
    assert status["fee"] == pytest.approx(2.0 * 101.0 * 0.001)


def test_order_status_does_not_apply_slippage_when_fill_price_is_final() -> None:
    exchange = SimulatedExchange(commission=0.001, slippage=0.01, execution_price_source=None)
    order = Order(
        "BTC/USDT",
        OrderSide.SELL,
        amount=2.0,
        price=95.0,
        meta={ORDER_META_FILL_PRICE_FINAL: True},
    )

    order_id = asyncio.run(exchange.place_order(order))
    status = asyncio.run(exchange.get_order_status(order_id))

    assert status["avg_price"] == pytest.approx(95.0)
    assert status["fee"] == pytest.approx(2.0 * 95.0 * 0.001)


def test_submit_order_lifecycle_returns_accepted_and_fill_events() -> None:
    exchange = SimulatedExchange(
        commission=0.001,
        slippage=0.01,
        order_id_prefix="sim",
        execution_price_source=None,
    )
    order = Order("BTC/USDT", OrderSide.BUY, amount=2.0, price=100.0, client_order_id="client-1")
    command = PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))

    events = asyncio.run(exchange.submit_order_lifecycle(command))

    assert len(events) == 2
    assert isinstance(events[0], OrderAcceptedEvent)
    assert isinstance(events[1], FillEvent)
    assert order.id == "sim_1"
    assert events[0].order_id == "sim_1"
    assert events[1].order_id == "sim_1"
    assert events[1].client_order_id == "client-1"
    assert events[1].price == pytest.approx(101.0)
    assert events[1].fee == pytest.approx(0.202)
    assert events[1].command_reason == "ENTRY"


def test_cancel_order_lifecycle_reports_success_and_unknown_order_failure() -> None:
    exchange = SimulatedExchange(execution_price_source=None)
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0)
    order_id = asyncio.run(exchange.place_order(order))

    success = asyncio.run(exchange.cancel_order_lifecycle(CancelOrderCommand(order_id, reason="user")))
    failure = asyncio.run(exchange.cancel_order_lifecycle(CancelOrderCommand("missing", reason="user")))

    assert isinstance(success[0], OrderCancelledEvent)
    assert success[0].order_id == order_id
    assert success[0].reason == "user"
    assert isinstance(failure[0], OrderRejectedEvent)
    assert failure[0].reason == "cancel failed: missing"
