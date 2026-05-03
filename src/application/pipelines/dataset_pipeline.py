from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from core.config.dataset_config import DatasetConfig
from domain.ml.features import MarketContextAssembler, MasterFeatureBuilder
from domain.ml.labeling import LabelingConfig, finalize_feature_frame, triple_barrier_labeling
from domain.ml.labeling.barrier_policy import BarrierPolicy
from infrastructure.exchanges.bybit.bybit_service import BybitService
from infrastructure.persistence.parquet_writer import ParquetWriter

logger = logging.getLogger(__name__)


class DatasetPipeline:
    def __init__(self, cfg: DatasetConfig, labeling_cfg: LabelingConfig | None = None) -> None:
        self.cfg = cfg
        self.labeling_cfg = labeling_cfg or LabelingConfig.from_env()
        self._barrier_policy = BarrierPolicy(self.labeling_cfg)
        self.writer = ParquetWriter(cfg.output_dir)
        self.bybit = BybitService()
        self.feature_builder = MasterFeatureBuilder()
        self.context_assembler = MarketContextAssembler()

    def run(self) -> list[Path]:
        logger.info(
            "Dataset pipeline started | exchange=%s | symbols=%s | tf=%s | htf=%s | "
            "start_ts=%s | end_ts=%s | output_dir=%s",
            self.cfg.exchange,
            ", ".join(self.cfg.symbols),
            self.cfg.timeframe,
            self.cfg.htf_timeframe,
            self.cfg.start_ts_ms,
            self.cfg.end_ts_ms,
            self.cfg.output_dir,
        )
        logger.info(
            "Labeling config | horizon=%s | adaptive=%s[min=%s max=%s low=%.4f high=%.4f] | "
            "dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
            self.labeling_cfg.adaptive_horizon.base_horizon,
            self.labeling_cfg.adaptive_horizon.enabled,
            self.labeling_cfg.adaptive_horizon.min_horizon,
            self.labeling_cfg.adaptive_horizon.max_horizon,
            self.labeling_cfg.adaptive_horizon.vol_low,
            self.labeling_cfg.adaptive_horizon.vol_high,
            self.labeling_cfg.barrier.use_dynamic_barriers,
            self.labeling_cfg.barrier.min_pct,
            self.labeling_cfg.barrier.max_pct,
            self.labeling_cfg.barrier.tp_to_sl_ratio,
        )
        outputs: list[Path] = []
        base_candle_map: dict[str, pd.DataFrame] = {}
        htf_candle_map: dict[str, pd.DataFrame] = {}
        for symbol in self.cfg.symbols:
            logger.info("[%s] preparing candle maps...", symbol)
            base_frame = self._load_or_sync_klines(symbol)
            if base_frame.empty:
                logger.warning("[%s] no klines loaded, skipping symbol", symbol)
                continue
            self._warn_if_history_starts_late(symbol, base_frame, self.cfg.timeframe)
            logger.info("[%s] loading HTF klines (%s)...", symbol, self.cfg.htf_timeframe)
            htf_frame = self._load_or_sync_htf_klines(symbol)
            if htf_frame.empty:
                logger.warning("[%s] no HTF klines loaded for timeframe=%s", symbol, self.cfg.htf_timeframe)
                continue
            self._warn_if_history_starts_late(symbol, htf_frame, self.cfg.htf_timeframe)
            logger.info("[%s] loaded %s main klines and %s HTF klines", symbol, len(base_frame), len(htf_frame))
            base_candle_map[symbol] = self._attach_optional_market_context(symbol, base_frame)
            htf_candle_map[symbol] = htf_frame

        if not base_candle_map:
            logger.warning("No symbols with valid candle data, nothing to build")
            return outputs

        feature_result = self.feature_builder.build(base_candle_map, htf_candle_map)
        logger.info(
            "Feature build resolved | profile=%s | blocks=%s | features=%s",
            feature_result.profile_name,
            ", ".join(feature_result.active_blocks) or "none",
            len(feature_result.feature_columns),
        )
        for symbol in base_candle_map:
            logger.info("[%s] building labeled dataset...", symbol)
            frame = feature_result.feature_map.get(symbol, pd.DataFrame())
            if frame.empty:
                logger.warning("[%s] feature frame is empty, skipping symbol", symbol)
                continue
            logger.info("[%s] applying barrier columns...", symbol)
            frame = self._barrier_policy.apply_barriers_to_frame(frame)
            logger.info("[%s] applying triple barrier labeling...", symbol)
            frame = triple_barrier_labeling(frame, self.labeling_cfg)
            feature_columns = list(feature_result.feature_columns)
            logger.info("[%s] finalizing frame | features=%s", symbol, len(feature_columns))
            dataset = finalize_feature_frame(frame, feature_columns, self.labeling_cfg)
            symbol_key = symbol.replace("/", "")
            out = self.writer.write(
                dataset,
                f"labeled/source=bybit/symbol={symbol_key}/timeframe={self.cfg.timeframe}/dataset.parquet",
            )
            logger.info("[%s] saved %s rows -> %s", symbol, len(dataset), out)
            outputs.append(out)
        logger.info("Dataset pipeline finished | files=%s", len(outputs))
        return outputs

    def _attach_optional_market_context(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        contexts: dict[str, pd.DataFrame] = {}
        if self.cfg.include_open_interest:
            logger.info("[%s] loading open interest...", symbol)
            open_interest_df = self._load_or_sync_open_interest(symbol)
            open_interest_df = self.context_assembler.dedupe_rows(open_interest_df)
            if not open_interest_df.empty:
                contexts["open_interest"] = open_interest_df
                logger.info("[%s] attached open interest points=%s", symbol, len(open_interest_df))
            else:
                logger.info("[%s] open interest is empty", symbol)

        if self.cfg.include_funding:
            logger.info("[%s] loading funding rates...", symbol)
            funding_df = self._load_or_sync_funding(symbol)
            funding_df = self.context_assembler.dedupe_rows(funding_df)
            if not funding_df.empty:
                contexts["funding_rate"] = funding_df
                logger.info("[%s] attached funding points=%s", symbol, len(funding_df))
            else:
                logger.info("[%s] funding is empty", symbol)

        if self.cfg.include_premium_index:
            logger.info("[%s] loading premium index...", symbol)
            premium_index_df = self._load_or_sync_premium_index(symbol)
            premium_index_df = self.context_assembler.dedupe_rows(premium_index_df)
            if not premium_index_df.empty:
                contexts["premium_index_close"] = premium_index_df
                logger.info("[%s] attached premium index points=%s", symbol, len(premium_index_df))
            else:
                logger.info("[%s] premium index is empty", symbol)

        return self.context_assembler.attach_contexts(
            frame,
            contexts,
            add_missing_columns=False,
        )

    def _cache_file(self, symbol: str, suffix: str, timeframe: str | None = None) -> Path:
        symbol_key = symbol.replace("/", "")
        target_timeframe = timeframe or self.cfg.timeframe
        return Path(self.cfg.output_dir) / f"raw/source=bybit/symbol={symbol_key}/timeframe={target_timeframe}/{suffix}.parquet"

    def _align_to_next_candle_open(self, timestamp_ms: int, step_ms: int) -> int:
        remainder = timestamp_ms % step_ms
        if remainder == 0:
            return timestamp_ms
        return timestamp_ms + (step_ms - remainder)

    def _warn_if_history_starts_late(self, symbol: str, frame: pd.DataFrame, timeframe: str) -> None:
        if frame.empty or "timestamp" not in frame.columns:
            return
        timeframe_ms = self.bybit.get_timeframe_ms(timeframe)
        expected_first_open_ts = self._align_to_next_candle_open(self.cfg.start_ts_ms, timeframe_ms)
        first_ts = pd.to_datetime(frame["timestamp"].iloc[0], errors="coerce")
        if pd.isna(first_ts):
            return
        first_ts_ms = int(first_ts.timestamp() * 1000)
        if first_ts_ms <= expected_first_open_ts:
            return
        logger.warning(
            "[%s-%s] incomplete history: requested from %s UTC, expected first candle at %s UTC, "
            "but first available candle starts at %s UTC. The asset was likely listed after the requested start date.",
            symbol,
            timeframe,
            datetime.fromtimestamp(self.cfg.start_ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(expected_first_open_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(first_ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _is_cache_complete(self, frame: pd.DataFrame, step_ms: int) -> bool:
        if frame.empty:
            return False
        
        expected_first_ts_ms = self._align_to_next_candle_open(self.cfg.start_ts_ms, step_ms)
        expected_last_ts_ms = self.cfg.end_ts_ms - (self.cfg.end_ts_ms % step_ms)
        
        expected_first_ts = pd.to_datetime(expected_first_ts_ms, unit="ms", utc=True).tz_convert(None)
        expected_last_ts = pd.to_datetime(expected_last_ts_ms, unit="ms", utc=True).tz_convert(None)
        
        start_ts = pd.to_datetime(self.cfg.start_ts_ms, unit="ms", utc=True).tz_convert(None)
        end_ts = pd.to_datetime(self.cfg.end_ts_ms, unit="ms", utc=True).tz_convert(None)
        
        scoped = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)]
        if scoped.empty:
            return False
            
        first_ts = scoped["timestamp"].iloc[0]
        last_ts = scoped["timestamp"].iloc[-1]
        
        if first_ts > expected_first_ts or last_ts < expected_last_ts:
            return False
            
        # Allow up to 1% missing data to account for exchange-level gaps (e.g., zero volume periods)
        expected = int((expected_last_ts_ms - expected_first_ts_ms) // step_ms) + 1
        actual = scoped["timestamp"].nunique()
        return actual >= expected * 0.99

    def _load_or_sync_klines(self, symbol: str) -> pd.DataFrame:
        return self._load_or_sync_klines_by_timeframe(symbol, self.cfg.timeframe)

    def _load_or_sync_htf_klines(self, symbol: str) -> pd.DataFrame:
        return self._load_or_sync_klines_by_timeframe(symbol, self.cfg.htf_timeframe)

    def _load_or_sync_klines_by_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        cache_path = self._cache_file(symbol, "klines", timeframe=timeframe)
        timeframe_ms = self.bybit.get_timeframe_ms(timeframe)
        existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not existing.empty:
            existing = self.context_assembler.dedupe_rows(existing)
            if self._is_cache_complete(existing, timeframe_ms):
                logger.info("[%s-%s] klines cache fully covers requested range, skipping reload", symbol, timeframe)
                merged = existing
            else:
                last_ts_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
                fetch_start = max(self.cfg.start_ts_ms, last_ts_ms + timeframe_ms)
                if fetch_start <= self.cfg.end_ts_ms:
                    logger.info("[%s-%s] syncing missing klines from %s...", symbol, timeframe, fetch_start)
                    fresh = self.bybit.fetch_klines(symbol, timeframe, fetch_start, self.cfg.end_ts_ms)
                    fresh_df = self._klines_to_frame(symbol, fresh)
                    merged = pd.concat([existing, fresh_df], ignore_index=True)
                else:
                    logger.info("[%s-%s] kline cache has gaps in range, rebuilding from scratch", symbol, timeframe)
                    fresh = self.bybit.fetch_klines(symbol, timeframe, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
                    merged = self._klines_to_frame(symbol, fresh)
        else:
            logger.info("[%s-%s] no kline cache found, full download", symbol, timeframe)
            fresh = self.bybit.fetch_klines(symbol, timeframe, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
            merged = self._klines_to_frame(symbol, fresh)

        if merged.empty:
            return merged
        merged = self.context_assembler.dedupe_rows(merged)
        start_ts = pd.to_datetime(self.cfg.start_ts_ms, unit="ms", utc=True).tz_convert(None)
        end_ts = pd.to_datetime(self.cfg.end_ts_ms, unit="ms", utc=True).tz_convert(None)
        merged = merged[(merged["timestamp"] >= start_ts) & (merged["timestamp"] <= end_ts)].reset_index(drop=True)
        self.writer.write(merged, str(cache_path.relative_to(Path(self.cfg.output_dir))))
        return merged

    @staticmethod
    def _klines_to_frame(symbol: str, klines: list) -> pd.DataFrame:
        if not klines:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "symbol"])
        return pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(candle.open_time, unit="ms", utc=True).tz_convert(None),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "quote_volume": candle.quote_volume,
                    "symbol": symbol,
                }
                for candle in klines
            ]
        )

    def _load_or_sync_open_interest(self, symbol: str) -> pd.DataFrame:
        cache_path = self._cache_file(symbol, "open_interest")
        timeframe_ms = self.bybit.get_timeframe_ms(self.cfg.timeframe)
        existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not existing.empty:
            existing = self.context_assembler.dedupe_rows(existing)
            if self._is_cache_complete(existing, timeframe_ms):
                logger.info("[%s] open interest cache fully covers requested range, skipping reload", symbol)
                merged = existing
            else:
                last_ts_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
                fetch_start = max(self.cfg.start_ts_ms, last_ts_ms + timeframe_ms)
                if fetch_start <= self.cfg.end_ts_ms:
                    logger.info("[%s] syncing missing open interest from %s...", symbol, fetch_start)
                    fresh = self.bybit.fetch_open_interest(symbol, self.cfg.timeframe, fetch_start, self.cfg.end_ts_ms)
                    fresh_df = pd.DataFrame(
                        {
                            "timestamp": [pd.to_datetime(p.timestamp, unit="ms", utc=True).tz_convert(None) for p in fresh],
                            "open_interest": [p.open_interest for p in fresh],
                        }
                    )
                    merged = pd.concat([existing, fresh_df], ignore_index=True)
                else:
                    logger.info("[%s] open interest cache has gaps in range, rebuilding from scratch", symbol)
                    fresh = self.bybit.fetch_open_interest(symbol, self.cfg.timeframe, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
                    merged = pd.DataFrame(
                        {
                            "timestamp": [pd.to_datetime(p.timestamp, unit="ms", utc=True).tz_convert(None) for p in fresh],
                            "open_interest": [p.open_interest for p in fresh],
                        }
                    )
        else:
            logger.info("[%s] no open interest cache found, full download", symbol)
            fresh = self.bybit.fetch_open_interest(symbol, self.cfg.timeframe, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
            merged = pd.DataFrame(
                {
                    "timestamp": [pd.to_datetime(p.timestamp, unit="ms", utc=True).tz_convert(None) for p in fresh],
                    "open_interest": [p.open_interest for p in fresh],
                }
            )
        if merged.empty:
            return merged
        merged = self.context_assembler.dedupe_rows(merged)
        self.writer.write(merged, str(cache_path.relative_to(Path(self.cfg.output_dir))))
        return merged

    def _load_or_sync_funding(self, symbol: str) -> pd.DataFrame:
        cache_path = self._cache_file(symbol, "funding")
        funding_ms = self.bybit.get_funding_interval_ms(symbol)
        existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not existing.empty:
            existing = self.context_assembler.dedupe_rows(existing)
            if self._is_cache_complete(existing, funding_ms):
                logger.info("[%s] funding cache fully covers requested range, skipping reload", symbol)
                merged = existing
            else:
                last_ts_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
                fetch_start = max(self.cfg.start_ts_ms, last_ts_ms + funding_ms)
                if fetch_start <= self.cfg.end_ts_ms:
                    logger.info("[%s] syncing missing funding from %s...", symbol, fetch_start)
                    fresh = self.bybit.fetch_funding_rates(symbol, fetch_start, self.cfg.end_ts_ms)
                    fresh_df = pd.DataFrame(
                        {
                            "timestamp": [pd.to_datetime(p.funding_time, unit="ms", utc=True).tz_convert(None) for p in fresh],
                            "funding_rate": [p.funding_rate for p in fresh],
                        }
                    )
                    merged = pd.concat([existing, fresh_df], ignore_index=True)
                else:
                    logger.info("[%s] funding cache has gaps in range, rebuilding from scratch", symbol)
                    fresh = self.bybit.fetch_funding_rates(symbol, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
                    merged = pd.DataFrame(
                        {
                            "timestamp": [pd.to_datetime(p.funding_time, unit="ms", utc=True).tz_convert(None) for p in fresh],
                            "funding_rate": [p.funding_rate for p in fresh],
                        }
                    )
        else:
            logger.info("[%s] no funding cache found, full download", symbol)
            fresh = self.bybit.fetch_funding_rates(symbol, self.cfg.start_ts_ms, self.cfg.end_ts_ms)
            merged = pd.DataFrame(
                {
                    "timestamp": [pd.to_datetime(p.funding_time, unit="ms", utc=True).tz_convert(None) for p in fresh],
                    "funding_rate": [p.funding_rate for p in fresh],
                }
            )
        if merged.empty:
            return merged
        merged = self.context_assembler.dedupe_rows(merged)
        self.writer.write(merged, str(cache_path.relative_to(Path(self.cfg.output_dir))))
        return merged

    def _load_or_sync_premium_index(self, symbol: str) -> pd.DataFrame:
        cache_path = self._cache_file(symbol, "premium_index")
        timeframe_ms = self.bybit.get_timeframe_ms(self.cfg.timeframe)
        existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not existing.empty:
            if "premium_index" in existing.columns and "premium_index_close" not in existing.columns:
                existing = existing.rename(columns={"premium_index": "premium_index_close"})
            existing = self.context_assembler.dedupe_rows(existing)
            if self._is_cache_complete(existing, timeframe_ms):
                logger.info("[%s] premium index cache fully covers requested range, skipping reload", symbol)
                merged = existing
            else:
                last_ts_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
                fetch_start = max(self.cfg.start_ts_ms, last_ts_ms + timeframe_ms)
                if fetch_start <= self.cfg.end_ts_ms:
                    logger.info("[%s] syncing missing premium index from %s...", symbol, fetch_start)
                    fresh = self.bybit.fetch_premium_index_klines(symbol, self.cfg.timeframe, fetch_start, self.cfg.end_ts_ms)
                    fresh_df = pd.DataFrame(
                        {
                            "timestamp": [pd.to_datetime(candle.open_time, unit="ms", utc=True).tz_convert(None) for candle in fresh],
                            "premium_index_close": [candle.close for candle in fresh],
                        }
                    )
                    merged = pd.concat([existing, fresh_df], ignore_index=True)
                else:
                    logger.info("[%s] premium index cache has gaps in range, rebuilding from scratch", symbol)
                    fresh = self.bybit.fetch_premium_index_klines(
                        symbol,
                        self.cfg.timeframe,
                        self.cfg.start_ts_ms,
                        self.cfg.end_ts_ms,
                    )
                    merged = pd.DataFrame(
                        {
                            "timestamp": [
                                pd.to_datetime(candle.open_time, unit="ms", utc=True).tz_convert(None) for candle in fresh
                            ],
                            "premium_index_close": [candle.close for candle in fresh],
                        }
                    )
        else:
            logger.info("[%s] no premium index cache found, full download", symbol)
            fresh = self.bybit.fetch_premium_index_klines(
                symbol,
                self.cfg.timeframe,
                self.cfg.start_ts_ms,
                self.cfg.end_ts_ms,
            )
            merged = pd.DataFrame(
                {
                    "timestamp": [pd.to_datetime(candle.open_time, unit="ms", utc=True).tz_convert(None) for candle in fresh],
                    "premium_index_close": [candle.close for candle in fresh],
                }
            )
        if merged.empty:
            return merged
        merged = self.context_assembler.dedupe_rows(merged)
        self.writer.write(merged, str(cache_path.relative_to(Path(self.cfg.output_dir))))
        return merged
