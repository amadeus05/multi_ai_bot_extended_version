from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import requests

from core.types.event_record import EventRecord
from infrastructure.repositories.event_record_codec import record_to_row, row_to_record
from infrastructure.storage.supabase_connection import SupabaseConnection


class SupabaseEventRepository:
    def __init__(self, connection: SupabaseConnection, *, table: str = "trading_events") -> None:
        if not connection.url or not connection.service_key:
            raise ValueError("Supabase storage requires SUPABASE_URL and SUPABASE_SERVICE_KEY.")
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

    def _append_many_sync(self, records: list[EventRecord]) -> None:
        rows = [record_to_row(record) for record in records]
        response = requests.post(
            self._connection.rest_url(self._table),
            headers=self._connection.headers(prefer="resolution=ignore-duplicates"),
            json=rows,
            timeout=20,
        )
        response.raise_for_status()

    def _session_rows_sync(self, session_id: str) -> list[dict]:
        response = requests.get(
            self._connection.rest_url(self._table),
            headers=self._connection.headers(),
            params={
                "session_id": f"eq.{session_id}",
                "select": "event_id,session_id,sequence,ts,kind,source,payload_json",
                "order": "sequence.asc",
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
