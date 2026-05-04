from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalNotification:
    signal_id: int
    symbol: str
    side: str
    ts: pd.Timestamp
    entry_price: float
    amount: float
    stop_price: float | None
    take_price: float | None
    stop_pct: float | None
    take_pct: float | None
    p_long: float
    p_short: float
    signal_gap: float
    direction_prob: float
    proba_threshold: float
    min_signal_gap: float
    balance: float | None = None


@dataclass(frozen=True)
class TradeExitNotification:
    trade_number: int
    symbol: str
    side: str
    reason: str
    ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl_abs: float
    pnl_pct: float
    commission: float
    balance: float


@dataclass(frozen=True)
class SystemNotification:
    level: str
    title: str
    message: str
    where: str | None = None
    error: str | None = None
