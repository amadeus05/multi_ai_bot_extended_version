from dataclasses import dataclass
from typing import Union

import pandas as pd

from core.types.domain_types import Order, Tick


@dataclass(frozen=True)
class PlaceOrderCommand:
    order: Order
    reason: str | None = None
    ts: pd.Timestamp | None = None
    source_tick: Tick | None = None
    continue_with_entry: bool = False


@dataclass(frozen=True)
class CancelOrderCommand:
    order_id: str
    reason: str | None = None
    ts: pd.Timestamp | None = None


TradingCommand = Union[PlaceOrderCommand, CancelOrderCommand]
