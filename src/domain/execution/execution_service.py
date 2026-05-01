import uuid

from core.interfaces.exchange import Exchange
from core.types.domain_types import Order, Trade


class ExecutionService:
    async def execute(self, order: Order, exchange: Exchange) -> Trade:
        if order.client_order_id is None:
            order.client_order_id = str(uuid.uuid4())
        order_id = await exchange.place_order(order)
        order.id = order_id
        status = await exchange.get_order_status(order_id)
        return Trade(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            amount=float(status.get("filled", order.amount)),
            price=float(status.get("avg_price", status.get("price", order.price or 0.0))),
            fee=float(status.get("fee", 0.0)),
            meta=dict(order.meta) if getattr(order, "meta", None) else None,
        )
