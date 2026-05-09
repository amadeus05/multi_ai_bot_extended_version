from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Iterator

import pandas as pd

from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Tick


class HistoricalMultiSymbolReplayProvider(DataProvider):
    """Deterministic historical provider that replays one timestamp batch at a time."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        skip_initial_bars: int = 0,
    ) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
            out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if not out.empty:
                out["symbol"] = symbol
                self._frames[symbol] = out
        self._skip = max(0, int(skip_initial_bars))
        self._subs: dict[str, list[Callable[[Tick], Awaitable[None]]]] = defaultdict(list)
        self._cursor_by_symbol: dict[str, int] = {symbol: -1 for symbol in self._frames}

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        frame = self._frames.get(symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        bars = max(1, int(bars))
        cursor = self._cursor_by_symbol.get(symbol, -1)
        if cursor < 0:
            n = min(bars, len(frame))
            return frame.iloc[:n].copy()
        end = min(cursor, len(frame) - 1)
        start = max(0, end - bars + 1)
        return frame.iloc[start : end + 1].copy()

    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        self._subs[symbol].append(callback)

    async def run(self) -> None:
        for ticks in self.iter_replay_batches():
            for tick in ticks:
                for cb in list(self._subs.get(tick.symbol, [])):
                    maybe = cb(tick)
                    if asyncio.iscoroutine(maybe):
                        await maybe

    def iter_replay_batches(self) -> Iterator[list[Tick]]:
        rows: list[tuple[pd.Timestamp, str, int]] = []
        for symbol, frame in self._frames.items():
            start_idx = self._skip
            if start_idx >= len(frame):
                continue
            for idx in range(start_idx, len(frame) - 1):
                rows.append((pd.Timestamp(frame["timestamp"].iloc[idx + 1]), symbol, idx + 1))
        rows.sort(key=lambda item: (item[0], item[1]))

        batch: list[tuple[pd.Timestamp, str, int]] = []
        current_ts: pd.Timestamp | None = None
        for row in rows:
            ts = row[0]
            if current_ts is not None and ts != current_ts:
                yield self._rows_to_ticks(batch)
                batch = []
            current_ts = ts
            batch.append(row)
        if batch:
            yield self._rows_to_ticks(batch)

    def _rows_to_ticks(self, rows: list[tuple[pd.Timestamp, str, int]]) -> list[Tick]:
        for _, symbol, row_idx in rows:
            self._cursor_by_symbol[symbol] = row_idx - 1
        return [
            self._tick_from_row(symbol, self._frames[symbol].iloc[row_idx])
            for _, symbol, row_idx in rows
        ]

    @staticmethod
    def _tick_from_row(symbol: str, row) -> Tick:
        ts = pd.Timestamp(row["timestamp"])
        if hasattr(ts, "tz_convert") and ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        px = float(row.get("open", row.get("close", 0.0)))
        return Tick(
            symbol=symbol,
            ts=ts,
            bid=px,
            ask=px,
            price=px,
            volume=float(row.get("volume", 0.0)),
            open=float(row.get("open", px)),
            high=float(row.get("high", px)),
            low=float(row.get("low", px)),
            close=float(row.get("close", px)),
        )

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        values = [pd.Timestamp(frame["timestamp"].iloc[-1]) for frame in self._frames.values() if not frame.empty]
        return max(values) if values else None

    @property
    def last_closes(self) -> dict[str, float]:
        return {
            symbol: float(frame["close"].iloc[-1])
            for symbol, frame in self._frames.items()
            if not frame.empty
        }
