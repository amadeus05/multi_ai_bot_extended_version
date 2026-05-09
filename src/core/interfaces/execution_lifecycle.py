from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order
from core.types.events import TradingEvent


@runtime_checkable
class ExecutionLifecycleExchange(Protocol):
    """Optional exchange capability for event-native execution."""

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list[TradingEvent]:
        ...

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list[TradingEvent]:
        ...


@runtime_checkable
class MarketOrderPreparationExchange(Protocol):
    """Optional exchange capability for adjusting market-order references before sizing."""

    async def prepare_market_order(self, order: Order) -> None:
        ...
