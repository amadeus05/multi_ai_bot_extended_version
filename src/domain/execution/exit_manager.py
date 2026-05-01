from typing import Optional

from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.ml.labeling.barrier_target_labeling import resolve_trade_exit
from domain.ml.labeling.models import LabelingConfig, BarrierConfig, AdaptiveHorizonConfig


class ExitManager:
    """
    Отвечает за выходы из позиций по TP/SL.
    Использует ту же математику, что и barrier_target_labeling (Triple Barrier).
    В бэктесте вызывается с будущими барами (next_open, next_high, next_low) для каузальной проверки.
    В live будет вызываться по каждому тику (реализовано отдельно или через физические ордера).
    """

    def __init__(self, slippage: float) -> None:
        self._slippage = float(slippage)
        # Dummy config specifically to pass into resolve_trade_exit, 
        # since we only use it for the slippage parameter in that function.
        self._dummy_cfg = LabelingConfig(
            adaptive_horizon=AdaptiveHorizonConfig(enabled=False),
            barrier=BarrierConfig(
                atr_multiplier=1.0, 
                rvol_multiplier=1.0, 
                tp_to_sl_ratio=1.0, 
                slippage=self._slippage,
                stop_pct=0.01,
                take_pct=0.01,
            ),
            realized_vol_column=""
        )

    def check_causal_exit(
        self,
        position: Position,
        next_open: float,
        next_high: float,
        next_low: float,
        stop_pct: float,
        take_pct: float,
    ) -> tuple[Optional[float], Optional[str]]:
        """
        Проверяет, было ли пересечение TP/SL внутри следующей свечи.
        Returns: (exit_price, reason)
        """
        direction = 1 if position.side == PositionSide.LONG else -1
        return resolve_trade_exit(
            direction=direction,
            entry_price=position.entry_price,
            next_open=next_open,
            next_high=next_high,
            next_low=next_low,
            stop_pct=stop_pct,
            take_pct=take_pct,
            cfg=self._dummy_cfg,
        )
