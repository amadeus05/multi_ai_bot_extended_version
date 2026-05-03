from __future__ import annotations

import asyncio

from application.event_journal import EventJournal
from application.trading_engine import TradingEngine
from core.types.commands import TradingCommand
from core.types.events import TradingEvent


class TradingRuntimeLoop:
    """Sequential event loop around TradingEngine."""

    def __init__(self, engine: TradingEngine, *, journal: EventJournal | None = None) -> None:
        self._engine = engine
        self._journal = journal
        self._engine.set_event_journal(journal)
        self._queue: asyncio.Queue[TradingEvent | None] = asyncio.Queue()
        self._running = False

    async def publish(self, event: TradingEvent) -> None:
        await self._queue.put(event)

    async def stop(self) -> None:
        await self._queue.put(None)

    async def drain(self) -> None:
        await self._queue.join()

    def start(self) -> asyncio.Task[None]:
        return asyncio.create_task(self.run_forever())

    async def process_once(self, event: TradingEvent) -> list[TradingEvent]:
        if self._journal is not None:
            await self._journal.record_event(event, source="runtime")
        commands: list[TradingCommand] = await self._engine.process_event(event)
        return await self._engine.execute_commands(commands)

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            event = await self._queue.get()
            try:
                if event is None:
                    self._running = False
                    continue
                await self.process_once(event)
            finally:
                self._queue.task_done()
