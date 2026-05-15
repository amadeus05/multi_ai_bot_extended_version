from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    chart_path: Path | None = None


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
    entry_ts: pd.Timestamp | None = None
    balance_before: float | None = None
    winrate_pct: float | None = None
    stop_losses_count: int | None = None
    take_profits_count: int | None = None


@dataclass(frozen=True)
class SystemNotification:
    level: str
    title: str
    message: str
    where: str | None = None
    error: str | None = None
