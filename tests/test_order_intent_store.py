import asyncio

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from domain.execution.order_intent_store import InMemoryOrderIntentStore


def make_command(client_order_id: str = "client-1") -> PlaceOrderCommand:
    order = Order(
        "BTC/USDT",
        OrderSide.BUY,
        amount=1.0,
        price=100.0,
        client_order_id=client_order_id,
    )
    return PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))


def test_in_memory_store_records_pending_intent_once() -> None:
    store = InMemoryOrderIntentStore()
    command = make_command()

    first = asyncio.run(store.record_pending(command))
    second = asyncio.run(store.record_pending(command))
    intent = asyncio.run(store.get("client-1"))

    assert first is True
    assert second is False
    assert intent is not None
    assert intent.status == "pending"
    assert intent.symbol == "BTC/USDT"
    assert intent.side == "buy"
    assert intent.reason == "ENTRY"


def test_in_memory_store_marks_accepted_and_rejected() -> None:
    store = InMemoryOrderIntentStore()
    accepted_command = make_command("client-accepted")
    rejected_command = make_command("client-rejected")

    asyncio.run(store.record_pending(accepted_command))
    asyncio.run(store.record_pending(rejected_command))
    asyncio.run(store.mark_accepted("client-accepted", "order-1"))
    asyncio.run(store.mark_rejected("client-rejected", "min qty"))

    accepted = asyncio.run(store.get("client-accepted"))
    rejected = asyncio.run(store.get("client-rejected"))
    assert accepted.status == "accepted"
    assert accepted.order_id == "order-1"
    assert rejected.status == "rejected"
    assert rejected.reject_reason == "min qty"


def test_in_memory_store_lists_active_and_marks_terminal_states() -> None:
    store = InMemoryOrderIntentStore()
    asyncio.run(store.record_pending(make_command("client-pending")))
    asyncio.run(store.record_pending(make_command("client-filled")))
    asyncio.run(store.record_pending(make_command("client-cancelled")))
    asyncio.run(store.mark_accepted("client-filled", "order-filled"))
    asyncio.run(store.mark_filled("client-filled", "order-filled"))
    asyncio.run(store.mark_cancelled("client-cancelled", "user cancel"))

    active = asyncio.run(store.list_active())
    filled = asyncio.run(store.get("client-filled"))
    cancelled = asyncio.run(store.get("client-cancelled"))

    assert [intent.client_order_id for intent in active] == ["client-pending"]
    assert filled.status == "filled"
    assert filled.order_id == "order-filled"
    assert cancelled.status == "cancelled"
    assert cancelled.reject_reason == "user cancel"
