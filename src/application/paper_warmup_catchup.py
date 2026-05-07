from __future__ import annotations

import logging

import pandas as pd

from application.trading_engine import ENGINE_META_EXIT_REASON, ENGINE_META_REASON_SUFFIX, TradingEngine
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class PaperWarmupCatchup:
    """Replay warmup candles for restored paper positions, exit-only."""

    def __init__(
        self,
        *,
        portfolio: PortfolioManager,
        engine: TradingEngine,
        exit_manager: ExitManager,
    ) -> None:
        self._portfolio = portfolio
        self._engine = engine
        self._exit_manager = exit_manager

    async def run(self, frames: dict[str, pd.DataFrame]) -> None:
        positions = list(self._portfolio.get_open_positions())
        if not positions:
            return
        for position in positions:
            frame = frames.get(position.symbol)
            if frame is None or frame.empty:
                logger.warning("[%s] paper catch-up skipped | reason=no_warmup_frame", position.symbol)
                continue
            if not {"timestamp", "open", "high", "low", "close"}.issubset(frame.columns):
                logger.warning("[%s] paper catch-up skipped | reason=warmup_frame_missing_ohlc", position.symbol)
                continue
            if not position.meta:
                logger.warning("[%s] paper catch-up skipped | reason=position_meta_missing", position.symbol)
                continue
            if position.meta.get("barrier_stop_pct") is None or position.meta.get("barrier_take_pct") is None:
                logger.warning("[%s] paper catch-up skipped | reason=barriers_missing", position.symbol)
                continue

            entry_ts = pd.to_datetime(position.meta.get("entry_ts"))
            replay = frame.copy()
            replay["timestamp"] = pd.to_datetime(replay["timestamp"])
            replay = replay[replay["timestamp"] > entry_ts].sort_values("timestamp")
            if replay.empty:
                logger.info("[%s] paper catch-up skipped | reason=no_candles_after_entry | entry_ts=%s", position.symbol, entry_ts)
                continue

            logger.info(
                "[%s] paper catch-up started | entry_ts=%s | candles=%s",
                position.symbol,
                entry_ts,
                len(replay),
            )
            for _, row in replay.iterrows():
                tick = Tick(
                    symbol=position.symbol,
                    ts=pd.to_datetime(row["timestamp"]),
                    bid=float(row["close"]),
                    ask=float(row["close"]),
                    price=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                )
                events = await self._close_if_exit_hit(position, tick)
                if events:
                    logger.info("[%s] paper catch-up closed position | ts=%s | events=%s", position.symbol, tick.ts, len(events))
                    break
            else:
                logger.info("[%s] paper catch-up finished | result=position_still_open", position.symbol)

    async def _close_if_exit_hit(self, position, tick: Tick) -> list:
        meta = position.meta or {}
        exit_price, reason = self._exit_manager.check_causal_exit(
            position=position,
            next_open=float(tick.open if tick.open is not None else tick.price),
            next_high=float(tick.high if tick.high is not None else tick.price),
            next_low=float(tick.low if tick.low is not None else tick.price),
            stop_pct=float(meta["barrier_stop_pct"]),
            take_pct=float(meta["barrier_take_pct"]),
        )
        if exit_price is None:
            return []
        exit_side = OrderSide.SELL if position.side.value == "long" else OrderSide.BUY
        order = Order(
            symbol=position.symbol,
            side=exit_side,
            amount=position.amount,
            price=float(exit_price),
            meta={
                ORDER_META_FILL_PRICE_FINAL: True,
                ENGINE_META_EXIT_REASON: reason,
                ENGINE_META_REASON_SUFFIX: "CATCH-UP",
            },
        )
        command = PlaceOrderCommand(order, reason=reason, ts=tick.ts)
        events = await self._engine.execute_commands([command])
        logger.info(
            "[%s] paper catch-up exit hit | ts=%s | reason=%s | price=%.8f",
            position.symbol,
            tick.ts,
            reason,
            float(exit_price),
        )
        return events
