import uuid

import pandas as pd

from core.interfaces.exchange import Exchange
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Trade
from core.types.enums import OrderStatus
from core.types.events import FillEvent, OrderAcceptedEvent, OrderRejectedEvent, TradingEvent


class ExecutionService:
    async def execute(self, command: PlaceOrderCommand | Order, exchange: Exchange) -> list[TradingEvent]:
        order = command.order if isinstance(command, PlaceOrderCommand) else command
        command_ts = command.ts if isinstance(command, PlaceOrderCommand) else None
        if order.client_order_id is None:
            order.client_order_id = str(uuid.uuid4())
        order_id = await exchange.place_order(order)
        order.id = order_id
        status = await exchange.get_order_status(order_id)

        status_value = status.get("status")
        event_ts = command_ts or status.get("ts") or status.get("timestamp") or pd.Timestamp.utcnow()
        events: list[TradingEvent] = [
            OrderAcceptedEvent(
                ts=event_ts,
                order_id=order_id,
                client_order_id=order.client_order_id,
                order=order,
            )
        ]

        if status_value == OrderStatus.REJECTED or str(status_value).lower().endswith("rejected"):
            return [
                OrderRejectedEvent(
                    ts=events[0].ts,
                    client_order_id=order.client_order_id,
                    reason=str(status.get("reason", "rejected")),
                    order=order,
                )
            ]

        if status_value != OrderStatus.FILLED and not str(status_value).lower().endswith("filled"):
            return events

        trade = Trade(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            amount=float(status.get("filled", order.amount)),
            price=float(status.get("avg_price", status.get("price", order.price or 0.0))),
            fee=float(status.get("fee", 0.0)),
            ts=event_ts,
            meta=dict(order.meta) if getattr(order, "meta", None) else None,
        )
        events.append(
            FillEvent.from_trade(
                trade,
                client_order_id=order.client_order_id,
                command_reason=command.reason if isinstance(command, PlaceOrderCommand) else None,
                source_tick=command.source_tick if isinstance(command, PlaceOrderCommand) else None,
                continue_with_entry=command.continue_with_entry if isinstance(command, PlaceOrderCommand) else False,
            )
        )
        return events
