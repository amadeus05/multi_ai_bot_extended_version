from __future__ import annotations

import logging
from pathlib import Path

from core.types.domain_types import Order
from core.types.enums import OrderSide
from core.types.notifications import SignalNotification, TradeExitNotification
from domain.portfolio.portfolio_manager import PortfolioManager
from triple_barrier_chart import render_triple_barrier_chart


logger = logging.getLogger(__name__)

SIGNAL_CHART_PATH = Path("charts") / "paper_signal_chart.png"


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


class TradingNotificationFactory:
    def __init__(self, *, signal_chart_path: Path = SIGNAL_CHART_PATH) -> None:
        self._signal_chart_path = signal_chart_path

    async def signal(
        self,
        *,
        tick,
        order: Order,
        portfolio: PortfolioManager,
        features,
        signal_seq: int,
        include_chart: bool,
    ) -> SignalNotification:
        meta = order.meta or {}
        side = self._order_position_side(order)
        entry_price = float(order.price if order.price is not None else tick.price)
        stop_pct = _optional_float(meta.get("barrier_stop_pct"))
        take_pct = _optional_float(meta.get("barrier_take_pct"))
        stop_price = self._barrier_price(entry_price, side, stop_pct, is_take=False)
        take_price = self._barrier_price(entry_price, side, take_pct, is_take=True)
        direction_prob = float(meta.get("direction_prob", 0.0))
        chart_path = (
            await self._signal_chart_path_for(
                symbol=order.symbol,
                candles=features,
                entry_price=entry_price,
                stop_price=stop_price,
                take_price=take_price,
                probability=direction_prob,
            )
            if include_chart
            else None
        )
        return SignalNotification(
            signal_id=int(meta.get("signal_number", signal_seq)),
            symbol=order.symbol,
            side=side,
            ts=getattr(tick, "ts", None),
            entry_price=entry_price,
            amount=float(order.amount),
            stop_price=stop_price,
            take_price=take_price,
            stop_pct=stop_pct,
            take_pct=take_pct,
            p_long=float(meta.get("p_long", 0.0)),
            p_short=float(meta.get("p_short", 0.0)),
            signal_gap=float(meta.get("signal_gap", 0.0)),
            direction_prob=direction_prob,
            proba_threshold=float(meta.get("directional_proba_threshold", 0.0)),
            min_signal_gap=float(meta.get("min_signal_gap", 0.0)),
            balance=float(portfolio.cash.get("USDT", 0.0)),
            chart_path=chart_path,
        )

    @staticmethod
    def trade_exit(res: dict, portfolio: PortfolioManager) -> TradeExitNotification:
        closed = portfolio.closed_trade_results
        wins = sum(1 for row in closed if float(row.get("pnl_abs", 0.0)) > 0)
        tp_count = sum(1 for row in closed if str(row.get("reason", "")).upper() == "TP")
        sl_count = sum(1 for row in closed if str(row.get("reason", "")).upper() == "SL")
        total = len(closed)
        balance = float(portfolio.cash.get("USDT", 0.0))
        pnl_abs = float(res.get("pnl_abs", 0.0))
        return TradeExitNotification(
            trade_number=int(res.get("trade_number", 0)),
            symbol=str(res.get("symbol", "")),
            side=str(res.get("side", "")).upper(),
            reason=str(res.get("reason", "CLOSE")),
            ts=res.get("ts"),
            entry_price=float(res.get("entry_price", 0.0)),
            exit_price=float(res.get("exit_price", 0.0)),
            qty=float(res.get("qty", 0.0)),
            pnl_abs=pnl_abs,
            pnl_pct=float(res.get("pnl_pct", 0.0)),
            commission=float(res.get("commission", 0.0)),
            balance=balance,
            entry_ts=res.get("entry_ts"),
            balance_before=balance - pnl_abs,
            winrate_pct=(wins / total * 100.0) if total else None,
            stop_losses_count=sl_count,
            take_profits_count=tp_count,
        )

    @staticmethod
    def _order_position_side(order: Order) -> str:
        return "LONG" if order.side == OrderSide.BUY else "SHORT"

    @staticmethod
    def _barrier_price(entry_price: float, side: str, pct: float | None, *, is_take: bool) -> float | None:
        if pct is None:
            return None
        direction = 1.0 if side == "LONG" else -1.0
        sign = direction if is_take else -direction
        return float(entry_price) * (1.0 + sign * float(pct))

    async def _signal_chart_path_for(
        self,
        *,
        symbol: str,
        candles,
        entry_price: float,
        stop_price: float | None,
        take_price: float | None,
        probability: float,
    ) -> Path | None:
        if stop_price is None or take_price is None:
            return None

        try:
            if candles.empty:
                return None

            output_path = self._signal_chart_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_triple_barrier_chart(
                candles,
                entry=entry_price,
                stop=stop_price,
                take=take_price,
                probability=probability,
                output_path=output_path,
                title=symbol,
            )
            return output_path
        except Exception:
            logger.exception("Signal chart generation failed | symbol=%s", symbol)
            return None
