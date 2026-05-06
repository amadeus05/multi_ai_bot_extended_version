import asyncio

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderStatus
from domain.execution.order_intent_reconciler import OrderIntentReconciler
from domain.execution.order_intent_store import InMemoryOrderIntentStore


class StatusExchange:
    def __init__(self, statuses: dict[str, dict]) -> None:
        self.statuses = statuses
        self.queries: list[str] = []

    async def get_order_status(self, order_id: str) -> dict:
        self.queries.append(order_id)
        return self.statuses.get(order_id, {"status": OrderStatus.NEW})


def make_command(client_order_id: str) -> PlaceOrderCommand:
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0, client_order_id=client_order_id)
    return PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))


def test_order_intent_reconciler_marks_terminal_states() -> None:
    store = InMemoryOrderIntentStore()
    for client_order_id in ("client-filled", "client-cancelled", "client-rejected"):
        asyncio.run(store.record_pending(make_command(client_order_id)))
    exchange = StatusExchange(
        {
            "link:client-filled": {"status": OrderStatus.FILLED, "order_id": "order-filled"},
            "link:client-cancelled": {"status": OrderStatus.CANCELLED, "reason": "user cancel"},
            "link:client-rejected": {"status": OrderStatus.REJECTED, "reason": "min qty"},
        }
    )

    result = asyncio.run(OrderIntentReconciler(store=store, exchange=exchange).reconcile_active())

    filled = asyncio.run(store.get("client-filled"))
    cancelled = asyncio.run(store.get("client-cancelled"))
    rejected = asyncio.run(store.get("client-rejected"))
    assert result.checked == 3
    assert result.filled == 1
    assert result.cancelled == 1
    assert result.rejected == 1
    assert filled.status == "filled"
    assert filled.order_id == "order-filled"
    assert cancelled.status == "cancelled"
    assert cancelled.reject_reason == "user cancel"
    assert rejected.status == "rejected"
    assert rejected.reject_reason == "min qty"


def test_order_intent_reconciler_leaves_non_terminal_status_active_and_is_idempotent() -> None:
    store = InMemoryOrderIntentStore()
    asyncio.run(store.record_pending(make_command("client-open")))
    asyncio.run(store.record_pending(make_command("client-filled")))
    exchange = StatusExchange(
        {
            "link:client-open": {"status": OrderStatus.NEW, "order_id": "order-open"},
            "link:client-filled": {"status": OrderStatus.FILLED, "order_id": "order-filled"},
        }
    )

    first = asyncio.run(OrderIntentReconciler(store=store, exchange=exchange).reconcile_active())
    second = asyncio.run(OrderIntentReconciler(store=store, exchange=exchange).reconcile_active())

    open_intent = asyncio.run(store.get("client-open"))
    filled_intent = asyncio.run(store.get("client-filled"))
    assert first.checked == 2
    assert first.filled == 1
    assert second.checked == 1
    assert second.filled == 0
    assert open_intent.status == "pending"
    assert filled_intent.status == "filled"
    assert exchange.queries == [
        "link:client-open",
        "link:client-filled",
        "link:client-open",
    ]
