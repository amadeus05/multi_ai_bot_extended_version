from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

import pandas as pd

from application.event_journal import EventJournal
from application.trading_engine import TradingEngine
from application.trading_runtime_loop import TradingRuntimeLoop
from application.trading_state_restorer import NoopTradingStateRestorer, TradingStateRestorer
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model
from core.interfaces.notifier import Notifier
from core.types.domain_types import Tick
from core.types.events import MarketEvent, TradingEvent

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
        self._journal = journal
        self._data_provider = data_provider
        self._model = model
        self._symbols = symbols
        self._warmup_bars = warmup_bars if warmup_bars is not None else model.required_bars()
        self._state_restorer = state_restorer or NoopTradingStateRestorer(source="realtime")
        self._execution_event_source: Callable[[], AsyncIterator[TradingEvent]] | None = None
        self._pending_batch_ts: pd.Timestamp | None = None
        self._pending_batch: dict[str, Tick] = {}

    def set_state_restorer(self, state_restorer: TradingStateRestorer | None) -> None:
        self._state_restorer = state_restorer or NoopTradingStateRestorer(source="realtime")

    def set_execution_event_source(self, source: Callable[[], AsyncIterator[TradingEvent]] | None) -> None:
        self._execution_event_source = source

    async def restore_trading_state(self) -> None:
        if self._journal is not None:
            await self._journal.initialize()
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

    async def scan_warmup_snapshot(self) -> None:
        """Run one immediate scan using the latest warmed-up bar for visibility after startup."""
        ticks: list[Tick] = []
        for symbol in self._symbols:
            frame = await self._data_provider.warmup(symbol, self._warmup_bars)
            if frame.empty:
                logger.warning("[%s] initial scan skipped: warmup frame is empty", symbol)
                continue

            row = frame.iloc[-1]
            close = float(row.get("close", row.get("price", 0.0)))
            tick = Tick(
                symbol=symbol,
                ts=pd.Timestamp(row.get("timestamp")),
                bid=close,
                ask=close,
                price=close,
                volume=float(row.get("volume", 0.0)),
                open=float(row.get("open", close)),
                high=float(row.get("high", close)),
                low=float(row.get("low", close)),
                close=close,
            )
            ticks.append(tick)

        if not ticks:
            logger.warning("initial scan skipped: no warmed-up ticks")
            return

        logger.info("initial scan requested | symbols=%s", ", ".join(tick.symbol for tick in ticks))
        await self._runtime.process_market_batch(ticks)
        logger.info("initial scan finished | symbols=%s", len(ticks))

    @staticmethod
    def _batch_ts(tick: Tick) -> pd.Timestamp:
        ts = pd.Timestamp(tick.ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts

    async def _flush_market_batch(self) -> None:
        if not self._pending_batch:
            return
        ticks = [self._pending_batch[symbol] for symbol in sorted(self._pending_batch)]
        self._pending_batch = {}
        self._pending_batch_ts = None
        await self._runtime.publish_market_batch(ticks)

    async def _publish_tick(self, tick: Tick) -> None:
        ts = self._batch_ts(tick)
        if self._pending_batch_ts is not None and ts != self._pending_batch_ts:
            await self._flush_market_batch()
        self._pending_batch_ts = ts
        self._pending_batch[tick.symbol] = tick
        if set(self._pending_batch) >= set(self._symbols):
            await self._flush_market_batch()

    async def _run_execution_event_source(self) -> None:
        if self._execution_event_source is None:
            return
        async for event in self._execution_event_source():
            await self._flush_market_batch()
            await self._runtime.publish(event)

    async def run(self) -> None:
        await self.restore_trading_state()
        for symbol in self._symbols:
            self._data_provider.subscribe(symbol, self._publish_tick)
        await self.bootstrap()
        await self.scan_warmup_snapshot()

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
                await self._flush_market_batch()
                await self._runtime.drain()
            elif execution_source_task is not None and execution_source_task in done:
                execution_source_task.result()
                await self._flush_market_batch()
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
