from typing import Optional

from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.ml.labeling.barrier_target_labeling import resolve_trade_exit


class ExitManager:
    """
    Выход по TP/SL: каноническая логика ``resolve_trade_exit`` (triple-barrier).
    На бэктесте — следующий бар OHLC; в live/paper тик трактуется как open/high/low.
    """

    def __init__(self, slippage: float) -> None:
        self._slippage = float(slippage)

    def check_causal_exit(
        self,
        position: Position,
        next_open: float,
        next_high: float,
        next_low: float,
        stop_pct: float,
        take_pct: float,
    ) -> tuple[Optional[float], Optional[str]]:
        direction = 1 if position.side == PositionSide.LONG else -1
        return resolve_trade_exit(
            direction=direction,
            entry_price=position.entry_price,
            next_open=next_open,
            next_high=next_high,
            next_low=next_low,
            stop_pct=stop_pct,
            take_pct=take_pct,
            slippage=self._slippage,
        )
