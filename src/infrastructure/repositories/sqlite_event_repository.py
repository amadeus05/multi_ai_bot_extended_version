from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from core.types.event_record import EventRecord
from infrastructure.repositories.event_record_codec import record_to_row, row_to_record
from infrastructure.storage.sqlite_connection import SQLiteConnection


class SQLiteEventRepository:
    def __init__(self, connection: SQLiteConnection, *, table: str = "trading_events") -> None:
        self._connection = connection
        self._table = table

    async def append(self, record: EventRecord) -> None:
        await self.append_many([record])

    async def append_many(self, records: list[EventRecord]) -> None:
        if not records:
            return
        await asyncio.to_thread(self._append_many_sync, list(records))

    async def iter_session(self, session_id: str) -> AsyncIterator[EventRecord]:
        rows = await asyncio.to_thread(self._session_rows_sync, session_id)
        for row in rows:
            yield row_to_record(row)

    def ensure_schema(self) -> None:
        with self._connection.connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, sequence),
                    UNIQUE(session_id, event_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_session_seq
                ON {self._table}(session_id, sequence)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_kind
                ON {self._table}(kind)
                """
            )

    def _append_many_sync(self, records: list[EventRecord]) -> None:
        self.ensure_schema()
        rows = [record_to_row(record) for record in records]
        with self._connection.connect() as conn:
            conn.executemany(
                f"""
                INSERT OR IGNORE INTO {self._table}
                    (event_id, session_id, sequence, ts, kind, source, payload_json)
                VALUES
                    (:event_id, :session_id, :sequence, :ts, :kind, :source, :payload_json)
                """,
                rows,
            )

    def _session_rows_sync(self, session_id: str):
        self.ensure_schema()
        with self._connection.connect() as conn:
            return list(
                conn.execute(
                    f"""
                    SELECT event_id, session_id, sequence, ts, kind, source, payload_json
                    FROM {self._table}
                    WHERE session_id = ?
                    ORDER BY sequence ASC
                    """,
                    (session_id,),
                )
            )
