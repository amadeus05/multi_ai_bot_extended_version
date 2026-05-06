import asyncio

import pandas as pd
import pytest

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide, OrderStatus
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from domain.execution.execution_service import ExecutionService
from domain.execution.order_intent_store import InMemoryOrderIntentStore


class LegacyExchange:
    def __init__(self, status: dict | None = None, cancel_result: bool = True) -> None:
        self.status = status or {"status": OrderStatus.FILLED, "filled": 2.0, "avg_price": 101.0, "fee": 0.5}
        self.cancel_result = cancel_result
        self.placed_orders: list[Order] = []
        self.cancelled_orders: list[str] = []

    async def place_order(self, order: Order) -> str:
        self.placed_orders.append(order)
        return "legacy-order-1"

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled_orders.append(order_id)
        return self.cancel_result

    async def get_balance(self, asset: str) -> float:
        return 0.0

    async def get_positions(self) -> list:
        return []

    async def current_price(self, symbol: str) -> float:
        return 0.0

    async def get_order_status(self, order_id: str) -> dict:
        return dict(self.status)


class LifecycleExchange(LegacyExchange):
    def __init__(self) -> None:
        super().__init__()
        self.submitted_commands: list[PlaceOrderCommand] = []
        self.cancel_commands: list[CancelOrderCommand] = []

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list:
        self.submitted_commands.append(command)
        return [OrderAcceptedEvent(command.ts, "life-order-1", command.order.client_order_id, command.order)]

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list:
        self.cancel_commands.append(command)
        return [OrderCancelledEvent(command.ts, command.order_id, reason=command.reason)]


class FilledLifecycleExchange(LegacyExchange):
    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list:
        return [
            OrderAcceptedEvent(command.ts, "life-order-1", command.order.client_order_id, command.order),
            FillEvent(
                ts=command.ts,
                order_id="life-order-1",
                client_order_id=command.order.client_order_id,
                symbol=command.order.symbol,
                side=command.order.side,
                amount=command.order.amount,
                price=command.order.price or 0.0,
            ),
        ]

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list:
        return [OrderCancelledEvent(command.ts, command.order_id, reason=command.reason)]


def make_order() -> Order:
    return Order("BTC/USDT", OrderSide.BUY, amount=2.0, price=100.0, meta={"tag": "entry"})


def test_legacy_filled_order_emits_accepted_then_fill_with_command_metadata() -> None:
    ts = pd.Timestamp("2024-01-01T00:00:00")
    order = make_order()
    command = PlaceOrderCommand(order, reason="ENTRY", ts=ts, continue_with_entry=True)
    exchange = LegacyExchange(
        status={
            "status": OrderStatus.FILLED,
            "filled": 1.5,
            "avg_price": 101.0,
            "fee": 0.25,
            "execution_id": "exec-1",
        }
    )

    events = asyncio.run(ExecutionService().execute(command, exchange))

    assert order.client_order_id is not None
    assert order.id == "legacy-order-1"
    assert len(events) == 2
    assert isinstance(events[0], OrderAcceptedEvent)
    assert isinstance(events[1], FillEvent)
    assert events[0].event_id == "legacy-order-1:accepted"
    assert events[1].event_id == "legacy-order-1:fill:exec-1"
    assert events[1].amount == pytest.approx(1.5)
    assert events[1].price == pytest.approx(101.0)
    assert events[1].fee == pytest.approx(0.25)
    assert events[1].meta == {"tag": "entry"}
    assert events[1].command_reason == "ENTRY"
    assert events[1].continue_with_entry is True


def test_legacy_rejected_order_emits_only_rejection() -> None:
    order = make_order()
    exchange = LegacyExchange(status={"status": OrderStatus.REJECTED, "reason": "min qty"})

    events = asyncio.run(ExecutionService().execute(PlaceOrderCommand(order), exchange))

    assert len(events) == 1
    assert isinstance(events[0], OrderRejectedEvent)
    assert events[0].reason == "min qty"
    assert events[0].order is order


def test_legacy_cancel_success_and_failure_are_eventized() -> None:
    success_exchange = LegacyExchange(cancel_result=True)
    failure_exchange = LegacyExchange(cancel_result=False)

    success_events = asyncio.run(
        ExecutionService().execute(CancelOrderCommand("order-ok", reason="user"), success_exchange)
    )
    failure_events = asyncio.run(
        ExecutionService().execute(CancelOrderCommand("order-missing", reason="user"), failure_exchange)
    )

    assert isinstance(success_events[0], OrderCancelledEvent)
    assert success_events[0].order_id == "order-ok"
    assert success_events[0].reason == "user"
    assert isinstance(failure_events[0], OrderRejectedEvent)
    assert failure_events[0].reason == "cancel failed: order-missing"


def test_lifecycle_exchange_is_delegated_after_client_order_id_is_assigned() -> None:
    exchange = LifecycleExchange()
    order = make_order()
    command = PlaceOrderCommand(order, ts=pd.Timestamp("2024-01-01T00:00:00"))

    events = asyncio.run(ExecutionService().execute(command, exchange))

    assert order.client_order_id is not None
    assert exchange.submitted_commands == [command]
    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert events[0].client_order_id == order.client_order_id


def test_execution_service_assigns_deterministic_client_order_id_when_tick_context_exists() -> None:
    exchange = LifecycleExchange()
    order = make_order()
    tick = Tick(
        symbol="BTC/USDT",
        ts=pd.Timestamp("2024-01-01T00:00:00"),
        bid=100.0,
        ask=100.0,
        price=100.0,
        volume=1.0,
    )
    command = PlaceOrderCommand(order, reason="ENTRY", ts=tick.ts, source_tick=tick)

    asyncio.run(ExecutionService().execute(command, exchange))
    first_id = order.client_order_id

    second_order = make_order()
    second_command = PlaceOrderCommand(second_order, reason="ENTRY", ts=tick.ts, source_tick=tick)
    asyncio.run(ExecutionService().execute(second_command, exchange))

    assert first_id == second_order.client_order_id


def test_execution_service_intent_store_blocks_duplicate_submit() -> None:
    store = InMemoryOrderIntentStore()
    exchange = LifecycleExchange()
    service = ExecutionService(order_intent_store=store)
    order = make_order()
    order.client_order_id = "duplicate-client-id"
    command = PlaceOrderCommand(order, ts=pd.Timestamp("2024-01-01T00:00:00"))

    first_events = asyncio.run(service.execute(command, exchange))
    second_events = asyncio.run(service.execute(command, exchange))

    assert len(first_events) == 1
    assert second_events == []
    assert len(exchange.submitted_commands) == 1
    intent = asyncio.run(store.get("duplicate-client-id"))
    assert intent.status == "accepted"
    assert intent.order_id == "life-order-1"


def test_execution_service_intent_store_records_rejection_reason() -> None:
    store = InMemoryOrderIntentStore()
    service = ExecutionService(order_intent_store=store)
    order = make_order()
    order.client_order_id = "rejected-client-id"
    exchange = LegacyExchange(status={"status": OrderStatus.REJECTED, "reason": "min qty"})

    events = asyncio.run(service.execute(PlaceOrderCommand(order), exchange))

    assert isinstance(events[0], OrderRejectedEvent)
    intent = asyncio.run(store.get("rejected-client-id"))
    assert intent.status == "rejected"
    assert intent.reject_reason == "min qty"


def test_execution_service_intent_store_records_fill_terminal_state() -> None:
    store = InMemoryOrderIntentStore()
    service = ExecutionService(order_intent_store=store)
    order = make_order()
    order.client_order_id = "filled-client-id"

    events = asyncio.run(service.execute(PlaceOrderCommand(order), FilledLifecycleExchange()))

    assert isinstance(events[-1], FillEvent)
    intent = asyncio.run(store.get("filled-client-id"))
    assert intent.status == "filled"
    assert intent.order_id == "life-order-1"


def test_execution_service_without_intent_store_keeps_legacy_duplicate_behavior() -> None:
    exchange = LifecycleExchange()
    service = ExecutionService()
    order = make_order()
    order.client_order_id = "same-client-id"
    command = PlaceOrderCommand(order, ts=pd.Timestamp("2024-01-01T00:00:00"))

    asyncio.run(service.execute(command, exchange))
    asyncio.run(service.execute(command, exchange))

    assert len(exchange.submitted_commands) == 2
