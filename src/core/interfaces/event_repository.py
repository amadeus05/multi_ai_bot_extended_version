from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from core.types.event_record import EventRecord


class EventRepository(Protocol):
    async def append(self, record: EventRecord) -> None:
        ...

    async def append_many(self, records: list[EventRecord]) -> None:
        ...

    async def iter_session(self, session_id: str) -> AsyncIterator[EventRecord]:
        ...

    async def max_sequence(self, session_id: str) -> int:
        ...
