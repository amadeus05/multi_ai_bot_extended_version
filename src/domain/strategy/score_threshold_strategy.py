from typing import Optional

from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.strategy.signal_resolver import resolve_directional_signal
from domain.strategy.strategy_engine import Strategy


class BacktestParitySignalStrategy(Strategy):
    """Сигнал для backtest/paper/live: порог вероятности + минимальный gap между классами."""

    def __init__(
        self,
        *,
        directional_proba_threshold: float = 0.55,
        min_signal_gap: float = 0.05,
        allow_longs: bool = True,
        allow_shorts: bool = True,
        entry_amount: float = 1.0,
    ) -> None:
        self._directional_proba_threshold = float(directional_proba_threshold)
        self._min_signal_gap = float(min_signal_gap)
        self._allow_longs = bool(allow_longs)
        self._allow_shorts = bool(allow_shorts)
        self._entry_amount = float(entry_amount)


    def on_prediction(self, tick: Tick, prediction: dict, portfolio: PortfolioManager) -> Optional[Order]:
        p_long = float(prediction.get("p_long", 0.5))
        p_short = float(prediction.get("p_short", 0.5))
        signal, _, _ = resolve_directional_signal(
            p_long, 
            p_short,
            self._directional_proba_threshold,
            self._min_signal_gap
        )
        # Extract barriers if present
        meta = {}
        stop_pct = prediction.get("barrier_stop_pct")
        if stop_pct is not None:
            meta["barrier_stop_pct"] = float(stop_pct)
        take_pct = prediction.get("barrier_take_pct")
        if take_pct is not None:
            meta["barrier_take_pct"] = float(take_pct)
        
        # Add score
        meta["score"] = float(prediction.get("score", max(p_long, p_short)))
        meta["directional_proba_threshold"] = self._directional_proba_threshold
        meta["min_signal_gap"] = self._min_signal_gap

        if signal == 1:
            if not self._allow_longs:
                return None
            return Order(symbol=tick.symbol, side=OrderSide.BUY, amount=self._entry_amount, price=tick.price, meta=meta)
        if signal == -1:
            if not self._allow_shorts:
                return None
            return Order(symbol=tick.symbol, side=OrderSide.SELL, amount=self._entry_amount, price=tick.price, meta=meta)
        return None


# Backward compatibility
ScoreThresholdLongStrategy = BacktestParitySignalStrategy
