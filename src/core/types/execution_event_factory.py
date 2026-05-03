import pandas as pd

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order, Trade
from core.types.enums import OrderStatus
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent


class ExecutionEventFactory:
    @staticmethod
    def event_ts(command_ts, status: dict | None = None) -> pd.Timestamp:
        status = status or {}
        return command_ts or status.get("ts") or status.get("timestamp") or pd.Timestamp.utcnow()

    @staticmethod
    def is_status(status_value, expected: OrderStatus) -> bool:
        return status_value == expected or str(status_value).lower().endswith(str(expected.value).lower())

    @staticmethod
    def accepted(ts: pd.Timestamp, order_id: str, order: Order) -> OrderAcceptedEvent:
        return OrderAcceptedEvent(
            ts=ts,
            order_id=order_id,
            client_order_id=order.client_order_id,
            order=order,
        )

    @staticmethod
    def rejected(ts: pd.Timestamp, order: Order | None, reason: str) -> OrderRejectedEvent:
        return OrderRejectedEvent(
            ts=ts,
            client_order_id=order.client_order_id if order is not None else None,
            reason=reason,
            order=order,
        )

    @staticmethod
    def cancelled(
        ts: pd.Timestamp,
        order_id: str,
        *,
        client_order_id: str | None = None,
        reason: str | None = None,
    ) -> OrderCancelledEvent:
        return OrderCancelledEvent(
            ts=ts,
            order_id=order_id,
            client_order_id=client_order_id,
            reason=reason,
        )

    @staticmethod
    def fill_from_status(
        command: PlaceOrderCommand,
        order_id: str,
        status: dict,
        ts: pd.Timestamp,
    ) -> FillEvent:
        order = command.order
        trade = Trade(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            amount=float(status.get("filled", order.amount)),
            price=float(status.get("avg_price", status.get("price", order.price or 0.0))),
            fee=float(status.get("fee", 0.0)),
            ts=ts,
            meta=dict(order.meta) if getattr(order, "meta", None) else None,
        )
        return FillEvent.from_trade(
            trade,
            client_order_id=order.client_order_id,
            command_reason=command.reason,
            source_tick=command.source_tick,
            continue_with_entry=command.continue_with_entry,
        )

    @classmethod
    def cancel_rejected(cls, command: CancelOrderCommand, ts: pd.Timestamp) -> OrderRejectedEvent:
        return cls.rejected(ts, None, f"cancel failed: {command.order_id}")
