from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from core.types.enums import OrderSide, OrderType, PositionSide


@dataclass(frozen=True)
class Tick:
    symbol: str
    ts: pd.Timestamp
    bid: float
    ask: float
    price: float
    volume: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None


@dataclass
class Order:
    symbol: str
    side: OrderSide
    amount: float
    price: Optional[float] = None
    type: OrderType = OrderType.MARKET
    id: Optional[str] = None
    client_order_id: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


@dataclass
class Position:
    symbol: str
    side: PositionSide
    amount: float
    entry_price: float
    meta: Optional[dict[str, Any]] = None


@dataclass
class Trade:
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    fee: float = 0.0
    ts: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    meta: Optional[dict[str, Any]] = None
