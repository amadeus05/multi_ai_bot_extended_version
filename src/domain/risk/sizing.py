"""
Сайзинг под triple-barrier: риск на сделку, stop_pct, плечо (эталон — BacktestEngine).
"""
from __future__ import annotations


def compute_barrier_position_notional(
    balance_for_risk: float,
    effective_risk: float,
    stop_pct: float,
    leverage: float,
) -> tuple[float, float]:
    """
    Как в бэктесте: risk_capital = balance * risk; notional = min(risk_capital/stop, balance * lev).
    Возвращает (position_notional, required_margin).
    """
    b = float(balance_for_risk)
    er = float(effective_risk)
    sp = float(stop_pct)
    lev = max(1e-9, float(leverage))
    if b <= 0 or er < 0 or sp <= 0 or not (sp == sp):
        return 0.0, 0.0
    risk_capital = b * er
    position_notional = min(risk_capital / sp, b * lev)
    required_margin = position_notional / lev
    return float(position_notional), float(required_margin)


def cap_notional_to_available_margin(
    position_notional: float,
    required_margin: float,
    available_margin: float,
    leverage: float,
) -> tuple[float, float]:
    """Ограничение как в бэктесте: margin <= available, notional <= margin * lev."""
    lev = max(1e-9, float(leverage))
    req = min(float(required_margin), float(max(0.0, available_margin)))
    notional = min(float(position_notional), req * lev)
    return float(notional), float(req)
