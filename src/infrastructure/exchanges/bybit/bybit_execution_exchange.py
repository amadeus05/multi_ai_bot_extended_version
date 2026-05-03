from __future__ import annotations

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.interfaces.exchange import Exchange
from core.types.domain_types import Order, Position
from core.types.events import TradingEvent
from infrastructure.exchanges.bybit.bybit_adapter import BybitAdapter
from infrastructure.exchanges.bybit.bybit_mapper import BybitMapper


class BybitExecutionExchange(Exchange):
    """Live Bybit execution adapter.

    The class owns the private execution contract used by TradingEngine.
    Real signed REST/WebSocket execution is intentionally not implemented in
    this skeleton, so live mode fails early with an explicit message instead
    of using the historical market-data service as an exchange.
    """

    def __init__(
        self,
        api_key: str,
        secret: str,
        *,
        testnet: bool = False,
        adapter: BybitAdapter | None = None,
        mapper: BybitMapper | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret = secret
        self._testnet = bool(testnet)
        self.adapter = adapter or BybitAdapter()
        self.mapper = mapper or BybitMapper()

    def _not_implemented(self, method: str) -> RuntimeError:
        return RuntimeError(
            f"Bybit live execution is not implemented yet: {method}. "
            "Use paper/backtest mode, or implement signed Bybit execution before running live."
        )

    async def place_order(self, order: Order) -> str:
        raise self._not_implemented("place_order")

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list[TradingEvent]:
        raise self._not_implemented("submit_order_lifecycle")

    async def cancel_order(self, order_id: str) -> None:
        raise self._not_implemented("cancel_order")

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list[TradingEvent]:
        raise self._not_implemented("cancel_order_lifecycle")

    async def get_balance(self, asset: str) -> float:
        raise self._not_implemented("get_balance")

    async def get_positions(self) -> list[Position]:
        raise self._not_implemented("get_positions")

    async def current_price(self, symbol: str) -> float:
        raise self._not_implemented("current_price")

    async def get_order_status(self, order_id: str) -> dict:
        raise self._not_implemented("get_order_status")
