from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config.storage_config import StorageSettings
from core.types.event_record import EventRecord
from infrastructure.storage.storage_factory import build_event_repository


async def main() -> None:
    db_path = Path("data") / "smoke_trading_runtime.sqlite"
    if db_path.exists():
        db_path.unlink()

    repo = build_event_repository(
        StorageSettings(
            driver="sqlite",
            sqlite_path=str(db_path),
            events_table="trading_events",
        )
    )
    await repo.append(
        EventRecord(
            event_id="evt-1",
            session_id="session-1",
            sequence=1,
            ts=pd.Timestamp("2024-01-01T00:00:00Z"),
            kind="MarketEvent",
            source="smoke",
            payload={"symbol": "BTC/USDT", "price": 100.0},
        )
    )
    await repo.append(
        EventRecord(
            event_id="evt-1",
            session_id="session-1",
            sequence=1,
            ts=pd.Timestamp("2024-01-01T00:00:00Z"),
            kind="MarketEvent",
            source="smoke",
            payload={"symbol": "BTC/USDT", "price": 100.0},
        )
    )
    await repo.append(
        EventRecord(
            event_id="evt-2",
            session_id="session-1",
            sequence=2,
            ts=pd.Timestamp("2024-01-01T01:00:00Z"),
            kind="FillEvent",
            source="smoke",
            payload={"order_id": "sim_1", "amount": 1.0},
        )
    )

    rows = [record async for record in repo.iter_session("session-1")]
    assert [row.event_id for row in rows] == ["evt-1", "evt-2"], rows
    assert rows[0].payload["symbol"] == "BTC/USDT", rows[0]
    assert rows[1].payload["order_id"] == "sim_1", rows[1]

    print("storage repository smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
