import asyncio
import logging
import queue
from collections import defaultdict
from typing import Awaitable, Callable

import pandas as pd

from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Tick
from infrastructure.data_providers.bybit_kline_ticks import tick_from_kline_event
from infrastructure.exchanges.bybit.bybit_kline_stream import BybitKlineStream
from infrastructure.exchanges.bybit.bybit_mapper import BybitMapper

logger = logging.getLogger(__name__)


class KlineHeartbeatProvider(DataProvider):
    """Lightweight public kline stream for execution heartbeats."""

    def __init__(self, ws_url: str, timeframe: str = "1m") -> None:
        self._ws_url = ws_url
        self._timeframe = timeframe
        self._subs: dict[str, list[Callable[[Tick], Awaitable[None]]]] = defaultdict(list)
        self._mapper = BybitMapper()
        self._stream = BybitKlineStream(url=ws_url)
        self._api_to_symbol: dict[str, str] = {}

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        return pd.DataFrame()

    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        normalized = self._mapper.normalize_symbol(symbol)
        api_symbol = self._mapper.to_api_symbol(normalized)
        self._subs[normalized].append(callback)
        self._api_to_symbol[api_symbol] = normalized

    async def run(self) -> None:
        interval = self._mapper.to_interval(self._timeframe)
        topics = {f"kline.{interval}.{api_symbol}" for api_symbol in self._api_to_symbol}
        self._stream.sync_topics(topics)
        self._stream.start()
        if not self._stream.wait_until_connected(timeout=15.0):
            raise RuntimeError("Bybit heartbeat websocket connection timeout")

        loop = asyncio.get_running_loop()
        logger.info(
            "KlineHeartbeatProvider running | symbols=%s | timeframe=%s",
            ", ".join(self._subs.keys()),
            self._timeframe,
        )
        while True:
            try:
                event = await loop.run_in_executor(None, self._stream.get_event, 1.0)
            except queue.Empty:
                continue
            symbol = self._api_to_symbol.get(event.symbol)
            if symbol is None:
                continue
            tick = tick_from_kline_event(symbol, event)
            callbacks = list(self._subs.get(symbol, []))
            for callback in callbacks:
                result = callback(tick)
                if asyncio.iscoroutine(result):
                    await result
