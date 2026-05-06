from dataclasses import dataclass
from typing import Any, Union

import pandas as pd

from core.types.domain_types import Order, Tick, Trade
from core.types.enums import OrderSide


@dataclass(frozen=True)
class MarketEvent:
    """Canonical market-data event consumed by TradingEngine."""

    tick: Tick

    @property
    def symbol(self) -> str:
        return self.tick.symbol

    @property
    def ts(self) -> pd.Timestamp:
        return self.tick.ts


@dataclass(frozen=True)
class ExitCheckEvent:
    """Market-data event used only for TP/SL checks."""

    tick: Tick

    @property
    def symbol(self) -> str:
        return self.tick.symbol

    @property
    def ts(self) -> pd.Timestamp:
        return self.tick.ts


@dataclass(frozen=True)
class OrderAcceptedEvent:
    ts: pd.Timestamp
    order_id: str
    client_order_id: str
    order: Order
    event_id: str | None = None


@dataclass(frozen=True)
class OrderRejectedEvent:
    ts: pd.Timestamp
    client_order_id: str | None
    reason: str
    order: Order | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class OrderCancelledEvent:
    ts: pd.Timestamp
    order_id: str
    client_order_id: str | None = None
    reason: str | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class FillEvent:
    """Canonical execution fill event consumed by TradingEngine."""

    ts: pd.Timestamp
    order_id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    amount: float
    price: float
    fee: float = 0.0
    meta: dict[str, Any] | None = None
    command_reason: str | None = None
    source_tick: Tick | None = None
    continue_with_entry: bool = False
    event_id: str | None = None

    @classmethod
    def from_trade(
        cls,
        trade: Trade,
        client_order_id: str | None = None,
        *,
        command_reason: str | None = None,
        source_tick: Tick | None = None,
        continue_with_entry: bool = False,
        event_id: str | None = None,
    ) -> "FillEvent":
        return cls(
            ts=trade.ts,
            order_id=trade.order_id,
            client_order_id=client_order_id,
            symbol=trade.symbol,
            side=trade.side,
            amount=float(trade.amount),
            price=float(trade.price),
            fee=float(trade.fee),
            meta=dict(trade.meta) if trade.meta else None,
            command_reason=command_reason,
            source_tick=source_tick,
            continue_with_entry=continue_with_entry,
            event_id=event_id,
        )

    def to_trade(self) -> Trade:
        return Trade(
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            amount=float(self.amount),
            price=float(self.price),
            fee=float(self.fee),
            ts=self.ts,
            meta=dict(self.meta) if self.meta else None,
        )


TradingEvent = Union[
    MarketEvent,
    ExitCheckEvent,
    OrderAcceptedEvent,
    OrderRejectedEvent,
    OrderCancelledEvent,
    FillEvent,
]
