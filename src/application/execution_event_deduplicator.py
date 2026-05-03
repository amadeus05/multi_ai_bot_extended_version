from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent, TradingEvent


@dataclass
class ExecutionEventDeduplicator:
    max_seen: int = 10000
    _seen: set[str] = field(default_factory=set)
    _order: deque[str] = field(default_factory=deque)

    def should_process(self, event: TradingEvent) -> bool:
        event_id = self._event_id(event)
        if event_id is None:
            return True
        if event_id in self._seen:
            return False
        self._remember(event_id)
        return True

    def _remember(self, event_id: str) -> None:
        self._seen.add(event_id)
        self._order.append(event_id)
        while len(self._order) > self.max_seen:
            oldest = self._order.popleft()
            self._seen.discard(oldest)

    @staticmethod
    def _event_id(event: TradingEvent) -> str | None:
        explicit = getattr(event, "event_id", None)
        if explicit:
            return str(explicit)
        if isinstance(event, FillEvent):
            return ExecutionEventDeduplicator._fill_fingerprint(event)
        if isinstance(event, OrderAcceptedEvent):
            return f"{event.order_id}:accepted"
        if isinstance(event, OrderCancelledEvent):
            return f"{event.order_id}:cancelled"
        if isinstance(event, OrderRejectedEvent):
            return f"{event.client_order_id or 'unknown'}:rejected:{event.reason}"
        return None

    @staticmethod
    def _fill_fingerprint(event: FillEvent) -> str:
        fields: tuple[Any, ...] = (
            event.order_id,
            event.client_order_id,
            event.symbol,
            event.side.value,
            float(event.amount),
            float(event.price),
            float(event.fee),
            str(event.ts),
        )
        return "fill:" + ":".join(str(value) for value in fields)
