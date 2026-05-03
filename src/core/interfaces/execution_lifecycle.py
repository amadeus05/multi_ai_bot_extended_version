from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.events import TradingEvent


@runtime_checkable
class ExecutionLifecycleExchange(Protocol):
    """Optional exchange capability for event-native execution."""

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list[TradingEvent]:
        ...

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list[TradingEvent]:
        ...
