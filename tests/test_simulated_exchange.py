import asyncio

import pandas as pd
import pytest

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order, Position
from core.types.enums import OrderSide, PositionSide
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
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


def test_one_minute_exit_event_closes_long_position_on_take_profit() -> None:
    class ExitManager:
        def check_causal_exit(self, **_kwargs):
            return 102.0, "TP"

    event = type(
        "Kline",
        (),
        {
            "open_price": 100.0,
            "high_price": 103.0,
            "low_price": 99.0,
            "end_ms": 1_704_067_200_000,
        },
    )()
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    exchange = SimulatedExchange(
        commission=0.0,
        slippage=0.0,
        execution_price_source=None,
        portfolio=portfolio,
        exit_manager=ExitManager(),
    )

    fill = asyncio.run(exchange._exit_event_for_kline(portfolio.positions[0], event))

    assert isinstance(fill, FillEvent)
    assert fill.symbol == "BTC/USDT"
    assert fill.side == OrderSide.SELL
    assert fill.price == pytest.approx(102.0)
    assert fill.command_reason == "TP"
    assert fill.meta[ORDER_META_FILL_PRICE_FINAL] is True


def test_one_minute_exit_events_include_order_lifecycle() -> None:
    class ExitManager:
        def check_causal_exit(self, **_kwargs):
            return 98.0, "SL"

    event = type(
        "Kline",
        (),
        {
            "open_price": 100.0,
            "high_price": 101.0,
            "low_price": 97.0,
            "end_ms": 1_704_067_200_000,
        },
    )()
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    exchange = SimulatedExchange(
        commission=0.0,
        slippage=0.0,
        order_id_prefix="sim",
        execution_price_source=None,
        portfolio=portfolio,
        exit_manager=ExitManager(),
    )

    events = asyncio.run(exchange._exit_events_for_kline(portfolio.positions[0], event))

    assert len(events) == 2
    assert isinstance(events[0], OrderAcceptedEvent)
    assert isinstance(events[1], FillEvent)
    assert events[0].order_id == "sim_1"
    assert events[1].order_id == "sim_1"
    assert events[1].command_reason == "SL"


def test_one_minute_exit_stream_emits_full_order_lifecycle() -> None:
    class ExitManager:
        def check_causal_exit(self, **_kwargs):
            return 98.0, "SL"

    class FakeExitStream:
        def __init__(self) -> None:
            self.topics = set()
            self.stopped = False

        def start(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

        def wait_until_connected(self, timeout: float) -> bool:
            return True

        def sync_topics(self, topics) -> None:
            self.topics = set(topics)

        def get_event(self, timeout: float):
            return type(
                "Kline",
                (),
                {
                    "symbol": "BTCUSDT",
                    "open_price": 100.0,
                    "high_price": 101.0,
                    "low_price": 97.0,
                    "end_ms": 1_704_067_200_000,
                },
            )()

    async def collect_first_two_events(exchange: SimulatedExchange):
        stream = exchange.stream_execution_events()
        first = await anext(stream)
        second = await anext(stream)
        await stream.aclose()
        return first, second

    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    exchange = SimulatedExchange(
        commission=0.0,
        slippage=0.0,
        order_id_prefix="sim",
        execution_price_source=None,
        portfolio=portfolio,
        exit_manager=ExitManager(),
        execution=ExecutionService(),
    )
    fake_stream = FakeExitStream()
    exchange._exit_stream = fake_stream

    first, second = asyncio.run(collect_first_two_events(exchange))

    assert fake_stream.topics == {"kline.1.BTCUSDT"}
    assert fake_stream.stopped is True
    assert isinstance(first, OrderAcceptedEvent)
    assert isinstance(second, FillEvent)
    assert first.order_id == "sim_1"
    assert second.order_id == "sim_1"
    assert second.command_reason == "SL"
