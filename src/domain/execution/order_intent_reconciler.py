from __future__ import annotations

import logging
from dataclasses import dataclass

from core.interfaces.exchange import Exchange
from core.types.enums import OrderStatus
from core.types.execution_event_factory import ExecutionEventFactory
from domain.execution.order_intent_store import OrderIntentStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderIntentReconcileResult:
    checked: int = 0
    filled: int = 0
    cancelled: int = 0
    rejected: int = 0


class OrderIntentReconciler:
    def __init__(self, *, store: OrderIntentStore, exchange: Exchange) -> None:
        self._store = store
        self._exchange = exchange

    async def reconcile_active(self) -> OrderIntentReconcileResult:
        active = await self._store.list_active()
        result = OrderIntentReconcileResult(checked=len(active))
        filled = cancelled = rejected = 0
        for intent in active:
            status = await self._exchange.get_order_status(f"link:{intent.client_order_id}")
            status_value = status.get("status")
            order_id = str(status.get("order_id") or intent.order_id or "")
            reason = str(status.get("reason") or "")
            if ExecutionEventFactory.is_status(status_value, OrderStatus.FILLED):
                await self._store.mark_filled(intent.client_order_id, order_id or None)
                filled += 1
            elif ExecutionEventFactory.is_status(status_value, OrderStatus.CANCELLED):
                await self._store.mark_cancelled(intent.client_order_id, reason or "cancelled")
                cancelled += 1
            elif ExecutionEventFactory.is_status(status_value, OrderStatus.REJECTED):
                await self._store.mark_rejected(intent.client_order_id, reason or "rejected")
                rejected += 1
        result = OrderIntentReconcileResult(
            checked=len(active),
            filled=filled,
            cancelled=cancelled,
            rejected=rejected,
        )
        logger.info(
            "Order intents reconciled | checked=%s | filled=%s | cancelled=%s | rejected=%s",
            result.checked,
            result.filled,
            result.cancelled,
            result.rejected,
        )
        return result
