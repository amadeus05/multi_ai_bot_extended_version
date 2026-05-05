from __future__ import annotations

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.interfaces.exchange import Exchange
from core.types.domain_types import Order, Position
from core.types.events import TradingEvent
from core.types.execution_event_factory import ExecutionEventFactory
from infrastructure.exchanges.bybit.bybit_execution_mapper import BybitExecutionMapper
from infrastructure.exchanges.bybit.bybit_rest_client import BybitRestClient, BybitRestError


class BybitExecutionExchange(Exchange):
    """Live/testnet Bybit execution facade used by TradingEngine."""

    def __init__(
        self,
        api_key: str,
        secret: str,
        *,
        testnet: bool = False,
        category: str = "linear",
        client: BybitRestClient | None = None,
        mapper: BybitExecutionMapper | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret = secret
        self._testnet = bool(testnet)
        self.client = client or BybitRestClient(api_key, secret, testnet=testnet)
        self.mapper = mapper or BybitExecutionMapper(category=category)

    async def place_order(self, order: Order) -> str:
        events = await self.submit_order_lifecycle(PlaceOrderCommand(order))
        for event in events:
            order_id = getattr(event, "order_id", None)
            if order_id:
                return str(order_id)
        raise RuntimeError("Bybit order was not accepted")

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list[TradingEvent]:
        order = command.order
        try:
            payload = self.mapper.to_create_order_payload(command)
            response = self.client.post("/v5/order/create", payload, request_name="order_create")
        except (BybitRestError, ValueError) as exc:
            return [ExecutionEventFactory.rejected(ExecutionEventFactory.event_ts(command.ts), order, str(exc))]

        result = response.get("result") or {}
        order_id = str(result.get("orderId") or "")
        if not order_id:
            return [
                ExecutionEventFactory.rejected(
                    ExecutionEventFactory.event_ts(command.ts),
                    order,
                    "Bybit order_create response is missing orderId",
                )
            ]
        order.id = order_id
        if result.get("orderLinkId") and order.client_order_id is None:
            order.client_order_id = str(result["orderLinkId"])
        return [ExecutionEventFactory.accepted(ExecutionEventFactory.event_ts(command.ts), order_id, order)]

    async def cancel_order(self, order_id: str) -> None:
        payload = self.mapper.to_cancel_order_payload(order_id)
        self.client.post("/v5/order/cancel", payload, request_name="order_cancel")
        return True

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list[TradingEvent]:
        event_ts = ExecutionEventFactory.event_ts(command.ts)
        try:
            await self.cancel_order(command.order_id)
        except BybitRestError:
            return [ExecutionEventFactory.cancel_rejected(command, event_ts)]
        return [
            ExecutionEventFactory.cancelled(
                event_ts,
                command.order_id,
                reason=command.reason,
            )
        ]

    async def get_balance(self, asset: str) -> float:
        response = self.client.get(
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED", "coin": asset.upper()},
            request_name="wallet_balance",
        )
        coins = ((response.get("result") or {}).get("list") or [{}])[0].get("coin") or []
        for coin in coins:
            if str(coin.get("coin", "")).upper() == asset.upper():
                return float(coin.get("walletBalance") or coin.get("equity") or 0.0)
        return 0.0

    async def get_positions(self) -> list[Position]:
        response = self.client.get(
            "/v5/position/list",
            {"category": self.mapper.category, "settleCoin": "USDT"},
            request_name="position_list",
        )
        rows = (response.get("result") or {}).get("list") or []
        positions: list[Position] = []
        for row in rows:
            position = self.mapper.to_position(row)
            if position is not None:
                positions.append(position)
        return positions

    async def current_price(self, symbol: str) -> float:
        response = self.client.get(
            "/v5/market/tickers",
            {"category": self.mapper.category, "symbol": self.mapper.to_api_symbol(symbol)},
            signed=False,
            request_name="ticker",
        )
        rows = (response.get("result") or {}).get("list") or []
        if not rows:
            return 0.0
        return float(rows[0].get("lastPrice") or rows[0].get("markPrice") or 0.0)

    async def get_order_status(self, order_id: str) -> dict:
        params = {"category": self.mapper.category}
        if order_id.startswith("link:"):
            params["orderLinkId"] = order_id.removeprefix("link:")
        else:
            params["orderId"] = order_id

        response = self.client.get("/v5/order/realtime", params, request_name="order_realtime")
        rows = (response.get("result") or {}).get("list") or []
        if not rows:
            response = self.client.get("/v5/order/history", params, request_name="order_history")
            rows = (response.get("result") or {}).get("list") or []
        if not rows:
            return {"status": "rejected", "reason": "Bybit order not found", "order_id": order_id}
        return self.mapper.to_status(rows[0])
