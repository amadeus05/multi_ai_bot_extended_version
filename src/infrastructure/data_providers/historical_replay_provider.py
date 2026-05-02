from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Iterator

import pandas as pd

from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Tick


class HistoricalReplayProvider(DataProvider):
    """Deterministic one-symbol historical provider for TradingEngine backtests."""

    def __init__(
        self,
        symbol: str,
        frame: pd.DataFrame,
        *,
        skip_initial_bars: int = 0,
    ) -> None:
        self._symbol = symbol
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        self._frame = out
        self._skip = max(0, int(skip_initial_bars))
        self._subs: dict[str, list[Callable[[Tick], Awaitable[None]]]] = defaultdict(list)
        self._cursor_idx = -1

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        if symbol != self._symbol or self._frame.empty:
            return pd.DataFrame()
        bars = max(1, int(bars))
        if self._cursor_idx < 0:
            n = min(bars, len(self._frame))
            return self._frame.iloc[:n].copy()
        end = min(self._cursor_idx, len(self._frame) - 1)
        start = max(0, end - bars + 1)
        return self._frame.iloc[start : end + 1].copy()

    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        self._subs[symbol].append(callback)

    async def run(self) -> None:
        for tick in self.iter_replay_ticks():
            for cb in list(self._subs.get(self._symbol, [])):
                maybe = cb(tick)
                if asyncio.iscoroutine(maybe):
                    await maybe

    def iter_replay_ticks(self) -> Iterator[Tick]:
        if self._frame.empty:
            return
        start_idx = self._skip
        if start_idx >= len(self._frame):
            return
        for idx in range(start_idx, len(self._frame) - 1):
            self._cursor_idx = idx
            next_row = self._frame.iloc[idx + 1]
            ts = pd.Timestamp(next_row["timestamp"])
            if hasattr(ts, "tz_convert") and ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            px = float(next_row.get("open", next_row.get("close", 0.0)))
            yield Tick(
                symbol=self._symbol,
                ts=ts,
                bid=px,
                ask=px,
                price=px,
                volume=float(next_row.get("volume", 0.0)),
                open=float(next_row.get("open", px)),
                high=float(next_row.get("high", px)),
                low=float(next_row.get("low", px)),
                close=float(next_row.get("close", px)),
            )

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        if self._frame.empty:
            return None
        return pd.Timestamp(self._frame["timestamp"].iloc[-1])

    @property
    def last_close(self) -> float | None:
        if self._frame.empty:
            return None
        return float(self._frame["close"].iloc[-1])
