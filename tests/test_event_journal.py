import asyncio

import pandas as pd

from application.event_journal import EventJournal
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from infrastructure.repositories.sqlite_event_repository import SQLiteEventRepository
from infrastructure.storage.sqlite_connection import SQLiteConnection


def test_event_journal_continues_existing_session_sequence(tmp_path) -> None:
    db_path = tmp_path / "journal.sqlite"
    repo = SQLiteEventRepository(SQLiteConnection(str(db_path)))
    session_id = "paper-main"

    first = EventJournal(repo, session_id=session_id)
    asyncio.run(first.record_command(_entry_command(), source="test"))

    second = EventJournal(repo, session_id=session_id)
    asyncio.run(second.initialize())
    asyncio.run(second.record_command(_entry_command("client-2"), source="test"))

    rows = asyncio.run(_session_rows(repo, session_id))
    assert [row.sequence for row in rows] == [1, 2]


def _entry_command(client_order_id: str = "client-1") -> PlaceOrderCommand:
    return PlaceOrderCommand(
        Order("ETH/USDT", OrderSide.BUY, amount=1.0, price=100.0, client_order_id=client_order_id),
        reason="ENTRY",
        ts=pd.Timestamp("2024-01-01T00:00:00"),
    )


async def _session_rows(repo: SQLiteEventRepository, session_id: str):
    return [row async for row in repo.iter_session(session_id)]
