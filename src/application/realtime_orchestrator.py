from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

from application.event_journal import EventJournal
from application.trading_engine import TradingEngine
from application.trading_runtime_loop import TradingRuntimeLoop
from application.trading_state_restorer import NoopTradingStateRestorer, TradingStateRestorer
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model
from core.interfaces.notifier import Notifier
from core.types.domain_types import Tick
from core.types.events import ExitCheckEvent, MarketEvent, TradingEvent

logger = logging.getLogger(__name__)


class RealtimeOrchestrator:
    """Single orchestrator for both live and paper modes.

    It handles startup warmup (history bootstrap) and then delegates
    realtime processing to TradingEngine.
    """

    def __init__(
        self,
        engine: TradingEngine,
        data_provider: DataProvider,
        model: Model,
        symbols: list[str],
        warmup_bars: int | None = None,
        runtime: TradingRuntimeLoop | None = None,
        journal: EventJournal | None = None,
        notifier: Notifier | None = None,
        state_restorer: TradingStateRestorer | None = None,
    ) -> None:
        self._notifier = notifier
        self._runtime = runtime or TradingRuntimeLoop(engine, journal=journal, notifier=notifier)
        self._data_provider = data_provider
        self._model = model
        self._symbols = symbols
        self._warmup_bars = warmup_bars if warmup_bars is not None else model.required_bars()
        self._state_restorer = state_restorer or NoopTradingStateRestorer(source="realtime")
        self._execution_event_source: Callable[[], AsyncIterator[TradingEvent]] | None = None

    def set_state_restorer(self, state_restorer: TradingStateRestorer | None) -> None:
        self._state_restorer = state_restorer or NoopTradingStateRestorer(source="realtime")

    def set_execution_event_source(self, source: Callable[[], AsyncIterator[TradingEvent]] | None) -> None:
        self._execution_event_source = source

    async def restore_trading_state(self) -> None:
        result = await self._state_restorer.restore_trading_state()
        logger.info(
            "Realtime state restore finished | source=%s | restored=%s | positions=%s",
            result.source,
            result.restored,
            result.positions_count,
        )

    async def bootstrap(self) -> None:
        logger.info("Realtime bootstrap started | symbols=%s | warmup_bars=%s", ", ".join(self._symbols), self._warmup_bars)
        for symbol in self._symbols:
            frame = await self._data_provider.warmup(symbol, self._warmup_bars)
            if frame.empty:
                logger.warning("[%s] warmup frame is empty", symbol)
            else:
                logger.info("[%s] warmup loaded rows=%s", symbol, len(frame))
        logger.info("Realtime bootstrap finished")

    async def _publish_tick(self, tick: Tick) -> None:
        await self._runtime.publish(MarketEvent(tick))

    async def _publish_exit_check_tick(self, tick: Tick) -> None:
        await self._runtime.publish(ExitCheckEvent(tick))

    async def _run_execution_event_source(self) -> None:
        if self._execution_event_source is None:
            return
        async for event in self._execution_event_source():
            await self._runtime.publish(event)

    async def run(self) -> None:
        await self.restore_trading_state()
        await self.bootstrap()
        for symbol in self._symbols:
            self._data_provider.subscribe(symbol, self._publish_tick)
            self._data_provider.subscribe_exit_checks(symbol, self._publish_exit_check_tick)

        runtime_task = self._runtime.start()
        provider_task = asyncio.create_task(self._data_provider.run())
        execution_source_task = (
            asyncio.create_task(self._run_execution_event_source())
            if self._execution_event_source is not None
            else None
        )
        try:
            tasks = {runtime_task, provider_task}
            if execution_source_task is not None:
                tasks.add(execution_source_task)
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if provider_task in done:
                provider_task.result()
                await self._runtime.drain()
            elif execution_source_task is not None and execution_source_task in done:
                execution_source_task.result()
                await self._runtime.drain()
            else:
                runtime_task.result()
        except Exception as exc:
            if self._notifier is not None and provider_task.done() and not runtime_task.done():
                with suppress(Exception):
                    await self._notifier.notify_error("RealtimeOrchestrator.run", exc)
            raise
        finally:
            await self._runtime.stop()
            if not provider_task.done():
                provider_task.cancel()
                with suppress(asyncio.CancelledError):
                    await provider_task
            if execution_source_task is not None and not execution_source_task.done():
                execution_source_task.cancel()
                with suppress(asyncio.CancelledError):
                    await execution_source_task
            if not runtime_task.done():
                with suppress(asyncio.CancelledError):
                    await runtime_task
