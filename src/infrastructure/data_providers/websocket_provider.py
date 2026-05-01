import asyncio
import logging
import queue
from collections import defaultdict
from typing import Awaitable, Callable

import pandas as pd

from core.config.base import BaseConfig
from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Tick
from domain.ml.features import MasterFeatureBuilder
from infrastructure.exchanges.bybit.bybit_kline_stream import BybitKlineStream
from infrastructure.exchanges.bybit.bybit_mapper import BybitMapper
from infrastructure.exchanges.bybit.bybit_service import BybitService

logger = logging.getLogger(__name__)


class WebSocketProvider(DataProvider):
    def __init__(self, ws_url: str, timeframe: str = "1h", htf_timeframe: str = "4h") -> None:
        self._ws_url = ws_url
        self._timeframe = timeframe
        self._htf_timeframe = htf_timeframe
        self._subs: dict[str, list[Callable[[Tick], Awaitable[None]]]] = defaultdict(list)
        self._mapper = BybitMapper()
        self._service = BybitService()
        self._feature_builder = MasterFeatureBuilder()
        self._stream = BybitKlineStream(url=ws_url)
        self._history: dict[str, pd.DataFrame] = {}
        self._htf_history: dict[str, pd.DataFrame] = {}
        self._open_interest: dict[str, pd.DataFrame] = {}
        self._funding: dict[str, pd.DataFrame] = {}
        self._premium_index: dict[str, pd.DataFrame] = {}
        self._api_to_symbol: dict[str, str] = {}
        self._include_open_interest = BaseConfig.env_str("INCLUDE_OPEN_INTEREST", "1") == "1"
        self._include_funding = BaseConfig.env_str("INCLUDE_FUNDING", "1") == "1"
        self._include_premium_index = BaseConfig.env_str("INCLUDE_PREMIUM_INDEX", "0") == "1"

    def _normalize_symbol(self, symbol: str) -> str:
        return self._mapper.normalize_symbol(symbol)

    def _to_api_symbol(self, symbol: str) -> str:
        normalized = self._normalize_symbol(symbol)
        return self._mapper.to_api_symbol(normalized)

    @staticmethod
    def _kline_to_row(symbol: str, kline) -> dict:
        return {
            "timestamp": pd.to_datetime(kline.open_time, unit="ms", utc=True).tz_convert(None),
            "open": kline.open,
            "high": kline.high,
            "low": kline.low,
            "close": kline.close,
            "volume": kline.volume,
            "quote_volume": kline.quote_volume,
            "symbol": symbol,
        }

    @staticmethod
    def _event_to_row(symbol: str, event) -> dict:
        return {
            "timestamp": pd.to_datetime(event.start_ms, unit="ms", utc=True).tz_convert(None),
            "open": event.open_price,
            "high": event.high_price,
            "low": event.low_price,
            "close": event.close_price,
            "volume": event.volume,
            "quote_volume": 0.0,
            "symbol": symbol,
        }

    @staticmethod
    def _dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
        out = out.dropna(subset=["timestamp"])
        out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        return out

    @staticmethod
    def _merge_asof_context(base: pd.DataFrame, context: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if base.empty:
            return base
        out = base.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if context.empty or value_col not in context.columns:
            out[value_col] = 0.0
            return out
        ctx = context.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], errors="coerce").astype("datetime64[ns]")
        ctx = ctx.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        out = pd.merge_asof(out, ctx[["timestamp", value_col]], on="timestamp", direction="backward")
        out[value_col] = out[value_col].ffill().fillna(0.0)
        return out

    def _sync_open_interest(self, symbol: str, start_ms: int, end_ms: int) -> None:
        if not self._include_open_interest or start_ms > end_ms:
            return
        tf_ms = self._service.get_timeframe_ms(self._timeframe)
        existing = self._open_interest.get(symbol, pd.DataFrame())
        if not existing.empty:
            existing = self._dedupe_rows(existing)
            last_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ms + tf_ms)
        fresh = self._service.fetch_open_interest(symbol, self._timeframe, start_ms, end_ms) if start_ms <= end_ms else []
        fresh_df = pd.DataFrame(
            {
                "timestamp": [pd.to_datetime(p.timestamp, unit="ms", utc=True).tz_convert(None) for p in fresh],
                "open_interest": [p.open_interest for p in fresh],
            }
        )
        merged = pd.concat([existing, fresh_df], ignore_index=True) if not existing.empty else fresh_df
        self._open_interest[symbol] = self._dedupe_rows(merged).tail(6000).reset_index(drop=True)

    def _sync_funding(self, symbol: str, start_ms: int, end_ms: int) -> None:
        if not self._include_funding or start_ms > end_ms:
            return
        funding_ms = self._service.get_funding_interval_ms(symbol)
        existing = self._funding.get(symbol, pd.DataFrame())
        if not existing.empty:
            existing = self._dedupe_rows(existing)
            last_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ms + funding_ms)
        fresh = self._service.fetch_funding_rates(symbol, start_ms, end_ms) if start_ms <= end_ms else []
        fresh_df = pd.DataFrame(
            {
                "timestamp": [pd.to_datetime(p.funding_time, unit="ms", utc=True).tz_convert(None) for p in fresh],
                "funding_rate": [p.funding_rate for p in fresh],
            }
        )
        merged = pd.concat([existing, fresh_df], ignore_index=True) if not existing.empty else fresh_df
        self._funding[symbol] = self._dedupe_rows(merged).tail(6000).reset_index(drop=True)

    def _sync_premium_index(self, symbol: str, start_ms: int, end_ms: int) -> None:
        if not self._include_premium_index or start_ms > end_ms:
            return
        tf_ms = self._service.get_timeframe_ms(self._timeframe)
        existing = self._premium_index.get(symbol, pd.DataFrame())
        if not existing.empty:
            existing = self._dedupe_rows(existing)
            last_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ms + tf_ms)
        fresh = self._service.fetch_premium_index_klines(symbol, self._timeframe, start_ms, end_ms) if start_ms <= end_ms else []
        fresh_df = pd.DataFrame(
            {
                "timestamp": [pd.to_datetime(c.open_time, unit="ms", utc=True).tz_convert(None) for c in fresh],
                "premium_index_close": [c.close for c in fresh],
            }
        )
        merged = pd.concat([existing, fresh_df], ignore_index=True) if not existing.empty else fresh_df
        self._premium_index[symbol] = self._dedupe_rows(merged).tail(6000).reset_index(drop=True)

    def _sync_context(self, symbol: str, start_ms: int, end_ms: int) -> None:
        self._sync_open_interest(symbol, start_ms, end_ms)
        self._sync_funding(symbol, start_ms, end_ms)
        self._sync_premium_index(symbol, start_ms, end_ms)

    def _sync_htf(self, symbol: str, start_ms: int, end_ms: int) -> None:
        if start_ms > end_ms:
            return
        htf_ms = self._service.get_timeframe_ms(self._htf_timeframe)
        existing = self._htf_history.get(symbol, pd.DataFrame())
        if not existing.empty:
            existing = self._dedupe_rows(existing)
            last_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ms + htf_ms)
        fresh = self._service.fetch_klines(symbol, self._htf_timeframe, start_ms, end_ms) if start_ms <= end_ms else []
        fresh_df = pd.DataFrame([self._kline_to_row(symbol, candle) for candle in fresh])
        merged = pd.concat([existing, fresh_df], ignore_index=True) if not existing.empty else fresh_df
        self._htf_history[symbol] = self._dedupe_rows(merged).tail(6000).reset_index(drop=True)

    def _enrich_history(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame
        if self._include_open_interest:
            out = self._merge_asof_context(out, self._open_interest.get(symbol, pd.DataFrame()), "open_interest")
        if self._include_funding:
            out = self._merge_asof_context(out, self._funding.get(symbol, pd.DataFrame()), "funding_rate")
        if self._include_premium_index:
            out = self._merge_asof_context(out, self._premium_index.get(symbol, pd.DataFrame()), "premium_index_close")
        return out

    def _build_feature_frame(self, symbol: str, bars: int) -> pd.DataFrame:
        base_map: dict[str, pd.DataFrame] = {}
        htf_map: dict[str, pd.DataFrame] = {}
        for subscribed_symbol in self._subs.keys():
            main_frame = self._history.get(subscribed_symbol, pd.DataFrame())
            htf_frame = self._htf_history.get(subscribed_symbol, pd.DataFrame())
            if main_frame.empty or htf_frame.empty:
                continue
            base_map[subscribed_symbol] = self._enrich_history(subscribed_symbol, main_frame.copy())
            htf_map[subscribed_symbol] = htf_frame.copy()
        if symbol not in base_map or symbol not in htf_map:
            return pd.DataFrame()
        result = self._feature_builder.build(base_map, htf_map)
        frame = result.feature_map.get(symbol, pd.DataFrame())
        if frame.empty:
            return frame
        return frame.tail(bars).reset_index(drop=True)

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        normalized = self._normalize_symbol(symbol)
        current = self._history.get(normalized, pd.DataFrame())
        if not current.empty and len(current) >= bars and normalized in self._htf_history:
            frame = self._build_feature_frame(normalized, bars)
            if not frame.empty:
                return frame

        tf_ms = self._service.get_timeframe_ms(self._timeframe)
        htf_ms = self._service.get_timeframe_ms(self._htf_timeframe)
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ms = now_ms - max(1, bars + 10) * tf_ms
        htf_start_ms = now_ms - max(1, (bars // max(1, htf_ms // tf_ms)) + 20) * htf_ms
        klines = self._service.fetch_klines(normalized, self._timeframe, start_ms, now_ms)
        loaded = pd.DataFrame([self._kline_to_row(normalized, candle) for candle in klines])
        merged = pd.concat([current, loaded], ignore_index=True) if not current.empty else loaded
        merged = self._dedupe_rows(merged)
        self._history[normalized] = merged
        if merged.empty:
            logger.warning("[%s] warmup loaded empty dataframe", normalized)
            return merged
        self._sync_context(normalized, start_ms, now_ms)
        self._sync_htf(normalized, htf_start_ms, now_ms)
        frame = self._build_feature_frame(normalized, bars)
        if frame.empty:
            return self._enrich_history(normalized, merged.tail(bars).reset_index(drop=True))
        return frame

    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        normalized = self._normalize_symbol(symbol)
        self._subs[normalized].append(callback)
        self._api_to_symbol[self._to_api_symbol(normalized)] = normalized

    async def run(self) -> None:
        interval = self._mapper.to_interval(self._timeframe)
        topics = {f"kline.{interval}.{api_symbol}" for api_symbol in self._api_to_symbol}
        self._stream.sync_topics(topics)
        self._stream.start()
        if not self._stream.wait_until_connected(timeout=15.0):
            raise RuntimeError("Bybit websocket connection timeout")

        loop = asyncio.get_running_loop()
        logger.info("WebSocketProvider running | symbols=%s | timeframe=%s", ", ".join(self._subs.keys()), self._timeframe)
        while True:
            try:
                event = await loop.run_in_executor(None, self._stream.get_event, 1.0)
            except queue.Empty:
                continue
            symbol = self._api_to_symbol.get(event.symbol)
            if symbol is None:
                continue
            row = self._event_to_row(symbol, event)
            current = self._history.get(symbol, pd.DataFrame())
            updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True) if not current.empty else pd.DataFrame([row])
            updated = self._dedupe_rows(updated)
            history = updated.tail(4000).reset_index(drop=True)
            self._history[symbol] = history
            end_ms = int(pd.to_datetime(history["timestamp"].iloc[-1]).timestamp() * 1000)
            start_ms = end_ms - self._service.get_timeframe_ms(self._timeframe) * 5
            self._sync_context(symbol, start_ms, end_ms)
            self._sync_htf(symbol, end_ms - self._service.get_timeframe_ms(self._htf_timeframe) * 2, end_ms)
            tick = Tick(
                symbol=symbol,
                ts=pd.to_datetime(event.end_ms, unit="ms", utc=True).tz_convert(None),
                bid=event.close_price,
                ask=event.close_price,
                price=event.close_price,
                volume=event.volume,
            )
            callbacks = list(self._subs.get(symbol, []))
            for callback in callbacks:
                result = callback(tick)
                if asyncio.iscoroutine(result):
                    await result
