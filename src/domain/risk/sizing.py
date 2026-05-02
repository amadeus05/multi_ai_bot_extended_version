"""
Сайзинг под triple-barrier: риск на сделку, stop_pct, плечо — общее ядро для TradingEngine/backtest.

- compute_barrier_position_notional / cap_notional_to_available_margin — низкоуровневые шаги;
- barrier_nominal_under_margin_cap — оба шага подряд (как второй проход открытия в бэктесте и live Risk);
- estimated_margin_reserved_perps_linear — оценка занятой маржи под открытые позиции (live/paper).
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


def estimated_margin_reserved_perps_linear(
    open_positions: object,
    leverage: float,
) -> float:
    """
    Грубая зарезервированная маржа под открытые позиции (USDT‑margin linear),
    в духе бэктеста: сумма (qty * entry_price / leverage).
    Duck-typing: у позиции есть amount и entry_price.
    """
    lev = max(1e-9, float(leverage))
    total = 0.0
    for pos in open_positions:
        amt = getattr(pos, "amount", None)
        px = getattr(pos, "entry_price", None)
        if amt is None or px is None:
            continue
        total += abs(float(amt)) * abs(float(px)) / lev
    return float(total)


def barrier_nominal_under_margin_cap(
    total_cash_snapshot: float,
    available_margin: float,
    effective_risk: float,
    stop_pct: float,
    leverage: float,
) -> tuple[float, float]:
    """
    Два шага как в TradingEngine backtest (и как в PositionSizingRule):
    номинал от полного snapshot, затем ограничение по свободной марже.
    """
    n, m = compute_barrier_position_notional(
        total_cash_snapshot, effective_risk, stop_pct, leverage
    )
    return cap_notional_to_available_margin(
        n, m, available_margin, leverage
    )
