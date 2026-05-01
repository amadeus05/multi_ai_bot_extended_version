from dataclasses import dataclass

import pandas as pd

from core.types.domain_types import Order, Tick, Trade


@dataclass(frozen=True)
class TickEvent:
    tick: Tick


@dataclass(frozen=True)
class SignalEvent:
    order: Order


@dataclass(frozen=True)
class OrderFilledEvent:
    trade: Trade
    ts: pd.Timestamp
