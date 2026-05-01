"""
Единая политика triple-barrier: тот же расчёт, что при сборке датасета (attach_barrier_columns).
"""
from __future__ import annotations

import pandas as pd

from domain.ml.labeling.barrier_target_labeling import attach_barrier_columns
from domain.ml.labeling.models import LabelingConfig


class BarrierPolicy:
    """Фасад над конфигом лейблинга и расчётом барьеров для train/backtest/live."""

    def __init__(self, cfg: LabelingConfig) -> None:
        self._cfg = cfg

    @property
    def cfg(self) -> LabelingConfig:
        return self._cfg

    @property
    def slippage(self) -> float:
        return float(self._cfg.barrier.slippage)

    def barriers_for_last_row(self, df: pd.DataFrame | None) -> tuple[float, float] | None:
        """
        Барьеры для последней строки окна фич (как в parquet после пайплайна).
        При ошибке динамических барьеров — статические stop_pct / take_pct из cfg.barrier.
        """
        if df is None or df.empty:
            return None
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return None
        try:
            out = attach_barrier_columns(df.copy(), self._cfg)
        except (ValueError, KeyError):
            stop = float(self._cfg.barrier.stop_pct)
            take = float(self._cfg.barrier.take_pct)
            return float(stop), float(take)
        if out.empty:
            return None
        last = out.iloc[-1]
        try:
            sp = float(last.get("barrier_stop_pct"))
            tp = float(last.get("barrier_take_pct"))
        except (TypeError, ValueError):
            return None
        if sp <= 0 or not (sp == sp) or not (tp == tp):
            return None
        return float(sp), float(tp)
