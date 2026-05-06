import asyncio
from pathlib import Path

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from infrastructure.repositories.sqlite_order_intent_store import SQLiteOrderIntentStore
from infrastructure.storage.sqlite_connection import SQLiteConnection


def make_command(client_order_id: str = "client-1") -> PlaceOrderCommand:
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0, client_order_id=client_order_id)
    return PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))


def test_sqlite_order_intent_store_persists_duplicates_across_instances() -> None:
    db_path = Path(".temp_code") / "test_order_intents.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()

    first_store = SQLiteOrderIntentStore(SQLiteConnection(str(db_path)))
    second_store = SQLiteOrderIntentStore(SQLiteConnection(str(db_path)))

    assert asyncio.run(first_store.record_pending(make_command())) is True
    assert asyncio.run(second_store.record_pending(make_command())) is False

    asyncio.run(second_store.mark_accepted("client-1", "order-1"))
    intent = asyncio.run(first_store.get("client-1"))

    assert intent is not None
    assert intent.status == "accepted"
    assert intent.order_id == "order-1"
    assert intent.symbol == "BTC/USDT"
    assert intent.side == "buy"


def test_sqlite_order_intent_store_persists_rejection_reason() -> None:
    db_path = Path(".temp_code") / "test_order_intents_rejected.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()

    store = SQLiteOrderIntentStore(SQLiteConnection(str(db_path)))

    asyncio.run(store.record_pending(make_command("client-rejected")))
    asyncio.run(store.mark_rejected("client-rejected", "min qty"))
    intent = asyncio.run(store.get("client-rejected"))

    assert intent is not None
    assert intent.status == "rejected"
    assert intent.reject_reason == "min qty"


def test_sqlite_order_intent_store_lists_active_and_terminal_states() -> None:
    db_path = Path(".temp_code") / "test_order_intents_terminal.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()

    store = SQLiteOrderIntentStore(SQLiteConnection(str(db_path)))

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
    assert filled is not None
    assert filled.status == "filled"
    assert filled.order_id == "order-filled"
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.reject_reason == "user cancel"
