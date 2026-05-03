from __future__ import annotations

import asyncio
import uuid

from core.interfaces.event_repository import EventRepository
from core.types.commands import TradingCommand
from core.types.event_record import EventRecord
from core.types.events import TradingEvent
from application.event_serialization import (
    command_kind,
    command_record_id,
    event_kind,
    event_record_id,
    object_ts,
    serialize_command,
    serialize_event,
)


class EventJournal:
    def __init__(
        self,
        repository: EventRepository,
        *,
        session_id: str | None = None,
        source: str = "trading_runtime",
    ) -> None:
        self._repository = repository
        self.session_id = session_id or self._new_session_id()
        self.source = source
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def record_event(self, event: TradingEvent, *, source: str | None = None) -> None:
        await self._append(
            event_id=event_record_id(event),
            ts=object_ts(event),
            kind=event_kind(event),
            source=source or self.source,
            payload=serialize_event(event),
        )

    async def record_command(self, command: TradingCommand, *, source: str | None = None) -> None:
        await self._append(
            event_id=command_record_id(command),
            ts=object_ts(command),
            kind=command_kind(command),
            source=source or self.source,
            payload=serialize_command(command),
        )

    async def _append(self, *, event_id: str | None, ts, kind: str, source: str, payload: dict) -> None:
        async with self._lock:
            self._sequence += 1
            sequence = self._sequence
        record = EventRecord(
            event_id=event_id or f"{self.session_id}:{sequence}",
            session_id=self.session_id,
            sequence=sequence,
            ts=ts,
            kind=kind,
            source=source,
            payload=payload,
        )
        await self._repository.append(record)

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{uuid.uuid4().hex}"
