from collections import defaultdict
import inspect
import os
from typing import Awaitable, Callable

import pandas as pd

from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Tick


class HistoricalDataProvider(DataProvider):
    """
    Legacy-style multi-symbol replay: один глобальный список bar_ts, на каждый бар
    собирается batch тиков — по одному на символ, у кого есть строка с этим временем.

    Порядок символов внутри бара **крутится (round-robin)** по номеру бара. Иначе при
    ``max_open_positions=1`` первый символ в SYMBOLS забирает почти все входы, а
    остальные остаются с нулём сделок — это не баг данных, а приоритет очереди.

    warmup отдаёт только историю текущего символа (не смесь всех монет).
    """

    def __init__(self, data: pd.DataFrame, symbol_order: list[str] | None = None) -> None:
        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        df["symbol"] = df["symbol"].astype(str)

        available = sorted(df["symbol"].unique())
        if symbol_order:
            ordered = [s for s in symbol_order if s in set(available)]
            rest = [s for s in available if s not in ordered]
            sym_keys = ordered + sorted(rest)
        else:
            sym_keys = available

        self._symbol_order: list[str] = sym_keys
        self._by_symbol: dict[str, pd.DataFrame] = {}
        for sym in sym_keys:
            sub = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
            if not sub.empty:
                self._by_symbol[sym] = sub

        self._timestamps: list[pd.Timestamp] = sorted(df["timestamp"].unique())

        self._pointers: dict[str, int] = {s: 0 for s in self._by_symbol}
        self._last_row_index: dict[str, int] = {s: -1 for s in self._by_symbol}

        self._subs: dict[str, list[Callable[[Tick], Awaitable[None]]]] = defaultdict(list)
        self._tick_counts: defaultdict[str, int] = defaultdict(int)

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        if symbol not in self._by_symbol:
            return pd.DataFrame()
        sdf = self._by_symbol[symbol]
        end = self._last_row_index.get(symbol, -1)
        if end < 0:
            return pd.DataFrame()
        left = max(0, end - bars + 1)
        return sdf.iloc[left : end + 1].copy()

    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        self._subs[symbol].append(callback)

    def _try_advance_symbol_to_bar(self, sym: str, bar_ts: pd.Timestamp) -> Tick | None:
        if sym not in self._by_symbol:
            return None
        sdf = self._by_symbol[sym]
        p = self._pointers[sym]
        if p >= len(sdf):
            return None
        row_ts = pd.Timestamp(sdf.iloc[p]["timestamp"])
        if row_ts != bar_ts:
            return None
        self._last_row_index[sym] = p
        self._pointers[sym] = p + 1
        row = sdf.iloc[p]
        ts_out = row_ts.tz_convert(None) if row_ts.tzinfo is not None else row_ts
        return Tick(
            symbol=sym,
            ts=ts_out,
            bid=float(row.get("bid", row.get("close", 0.0))),
            ask=float(row.get("ask", row.get("close", 0.0))),
            price=float(row.get("close", 0.0)),
            volume=float(row.get("volume", 0.0)),
        )

    async def run(self) -> None:
        self._tick_counts = defaultdict(int)
        verify = os.getenv("BACKTEST_VERIFY_TICK_BATCHES", "").strip() in ("1", "true", "yes")
        rr_disable = os.getenv("BACKTEST_SYMBOL_ORDER_FIXED", "").strip() in ("1", "true", "yes")
        n_sym = len(self._symbol_order)
        for bar_i, bar_ts in enumerate(self._timestamps):
            bar_ts = pd.Timestamp(bar_ts)
            bar_batch: list[Tick] = []
            if rr_disable or n_sym == 0:
                scan_order = self._symbol_order
            else:
                offset = bar_i % n_sym
                scan_order = self._symbol_order[offset:] + self._symbol_order[:offset]
            for sym in scan_order:
                tick = self._try_advance_symbol_to_bar(sym, bar_ts)
                if tick is not None:
                    bar_batch.append(tick)
            for tick in bar_batch:
                self._tick_counts[tick.symbol] += 1
                for cb in self._subs.get(tick.symbol, []):
                    maybe_coro = cb(tick)
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
        if verify and self._tick_counts:
            counts = dict(sorted(self._tick_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            mx = max(self._tick_counts.values())
            mn = min(self._tick_counts.values())
            ratio = (mx / mn) if mn else 0.0
            print(
                "[HistoricalDataProvider] ticks per symbol: "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
            )
            print(f"[HistoricalDataProvider] min_ticks={mn} max_ticks={mx} max/min={ratio:.3f}")
