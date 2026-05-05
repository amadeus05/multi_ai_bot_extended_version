from core.interfaces.exchange import Exchange
from core.interfaces.execution_lifecycle import ExecutionLifecycleExchange
from core.types.commands import CancelOrderCommand, PlaceOrderCommand, TradingCommand
from core.types.domain_types import Order
from core.types.enums import OrderStatus
from core.types.events import TradingEvent
from core.types.execution_event_factory import ExecutionEventFactory
from domain.execution.client_order_id import ensure_client_order_id


class ExecutionService:
    async def execute(self, command: TradingCommand | Order, exchange: Exchange) -> list[TradingEvent]:
        if isinstance(command, Order):
            command = PlaceOrderCommand(command)
        if isinstance(command, PlaceOrderCommand):
            self.ensure_client_order_id(command.order, command)
            if isinstance(exchange, ExecutionLifecycleExchange):
                return await exchange.submit_order_lifecycle(command)
            return await self._place_order_legacy(command, exchange)
        if isinstance(command, CancelOrderCommand):
            if isinstance(exchange, ExecutionLifecycleExchange):
                return await exchange.cancel_order_lifecycle(command)
            return await self._cancel_order_legacy(command, exchange)
        return []

    @staticmethod
    def ensure_client_order_id(order: Order, command: PlaceOrderCommand | None = None) -> None:
        ensure_client_order_id(order, command)

    async def _place_order_legacy(self, command: PlaceOrderCommand, exchange: Exchange) -> list[TradingEvent]:
        order = command.order
        order_id = await exchange.place_order(order)
        order.id = order_id
        status = await exchange.get_order_status(order_id)

        status_value = status.get("status")
        event_ts = ExecutionEventFactory.event_ts(command.ts, status)
        if ExecutionEventFactory.is_status(status_value, OrderStatus.REJECTED):
            return [ExecutionEventFactory.rejected(event_ts, order, str(status.get("reason", "rejected")))]
        if ExecutionEventFactory.is_status(status_value, OrderStatus.CANCELLED):
            return [
                ExecutionEventFactory.cancelled(
                    event_ts,
                    order_id=order_id,
                    client_order_id=order.client_order_id,
                    reason=str(status.get("reason", "cancelled")),
                )
            ]

        events: list[TradingEvent] = [
            ExecutionEventFactory.accepted(event_ts, order_id, order)
        ]

        if not ExecutionEventFactory.is_status(status_value, OrderStatus.FILLED):
            return events

        events.append(ExecutionEventFactory.fill_from_status(command, order_id, status, event_ts))
        return events

    async def _cancel_order_legacy(self, command: CancelOrderCommand, exchange: Exchange) -> list[TradingEvent]:
        cancelled = await exchange.cancel_order(command.order_id)
        event_ts = ExecutionEventFactory.event_ts(command.ts)
        if cancelled is False:
            return [ExecutionEventFactory.cancel_rejected(command, event_ts)]
        return [
            ExecutionEventFactory.cancelled(
                event_ts,
                command.order_id,
                reason=command.reason,
            )
        ]
