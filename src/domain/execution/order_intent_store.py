from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from core.types.commands import PlaceOrderCommand


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    symbol: str
    side: str
    status: str
    reason: str | None = None
    order_id: str | None = None
    reject_reason: str | None = None
    ts: pd.Timestamp | None = None


class OrderIntentStore(Protocol):
    async def record_pending(self, command: PlaceOrderCommand) -> bool:
        ...

    async def mark_accepted(self, client_order_id: str, order_id: str) -> None:
        ...

    async def mark_rejected(self, client_order_id: str, reason: str) -> None:
        ...

    async def mark_cancelled(self, client_order_id: str, reason: str | None = None) -> None:
        ...

    async def mark_filled(self, client_order_id: str, order_id: str | None = None) -> None:
        ...

    async def get(self, client_order_id: str) -> OrderIntent | None:
        ...

    async def list_active(self) -> list[OrderIntent]:
        ...


class InMemoryOrderIntentStore:
    def __init__(self) -> None:
        self._intents: dict[str, OrderIntent] = {}

    async def record_pending(self, command: PlaceOrderCommand) -> bool:
        client_order_id = command.order.client_order_id
        if not client_order_id:
            raise ValueError("client_order_id is required before recording order intent")
        if client_order_id in self._intents:
            return False
        self._intents[client_order_id] = OrderIntent(
            client_order_id=client_order_id,
            symbol=command.order.symbol,
            side=command.order.side.value,
            status="pending",
            reason=command.reason,
            ts=command.ts,
        )
        return True

    async def mark_accepted(self, client_order_id: str, order_id: str) -> None:
        current = self._intents.get(client_order_id)
        if current is None:
            return
        self._intents[client_order_id] = OrderIntent(
            client_order_id=current.client_order_id,
            symbol=current.symbol,
            side=current.side,
            status="accepted",
            reason=current.reason,
            order_id=order_id,
            reject_reason=current.reject_reason,
            ts=current.ts,
        )

    async def mark_rejected(self, client_order_id: str, reason: str) -> None:
        current = self._intents.get(client_order_id)
        if current is None:
            return
        self._intents[client_order_id] = OrderIntent(
            client_order_id=current.client_order_id,
            symbol=current.symbol,
            side=current.side,
            status="rejected",
            reason=current.reason,
            order_id=current.order_id,
            reject_reason=reason,
            ts=current.ts,
        )

    async def mark_cancelled(self, client_order_id: str, reason: str | None = None) -> None:
        current = self._intents.get(client_order_id)
        if current is None:
            return
        self._intents[client_order_id] = OrderIntent(
            client_order_id=current.client_order_id,
            symbol=current.symbol,
            side=current.side,
            status="cancelled",
            reason=current.reason,
            order_id=current.order_id,
            reject_reason=reason,
            ts=current.ts,
        )

    async def mark_filled(self, client_order_id: str, order_id: str | None = None) -> None:
        current = self._intents.get(client_order_id)
        if current is None:
            return
        self._intents[client_order_id] = OrderIntent(
            client_order_id=current.client_order_id,
            symbol=current.symbol,
            side=current.side,
            status="filled",
            reason=current.reason,
            order_id=order_id or current.order_id,
            reject_reason=current.reject_reason,
            ts=current.ts,
        )

    async def get(self, client_order_id: str) -> OrderIntent | None:
        return self._intents.get(client_order_id)

    async def list_active(self) -> list[OrderIntent]:
        return [
            intent
            for intent in self._intents.values()
            if intent.status in {"pending", "accepted"}
        ]
