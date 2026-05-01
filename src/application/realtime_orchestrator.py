from __future__ import annotations

import logging

from application.trading_engine import TradingEngine
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model

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
    ) -> None:
        self._engine = engine
        self._data_provider = data_provider
        self._model = model
        self._symbols = symbols
        self._warmup_bars = warmup_bars if warmup_bars is not None else model.required_bars()

    async def bootstrap(self) -> None:
        logger.info("Realtime bootstrap started | symbols=%s | warmup_bars=%s", ", ".join(self._symbols), self._warmup_bars)
        for symbol in self._symbols:
            frame = await self._data_provider.warmup(symbol, self._warmup_bars)
            if frame.empty:
                logger.warning("[%s] warmup frame is empty", symbol)
            else:
                logger.info("[%s] warmup loaded rows=%s", symbol, len(frame))
        logger.info("Realtime bootstrap finished")

    async def run(self) -> None:
        await self.bootstrap()
        await self._engine.run(self._symbols)
