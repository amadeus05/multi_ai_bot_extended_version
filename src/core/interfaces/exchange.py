from abc import ABC, abstractmethod

from core.types.domain_types import Order, Position


class Exchange(ABC):
    async def prepare_market_order(self, order: Order) -> None:
        return None

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_balance(self, asset: str) -> float:
        raise NotImplementedError

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    async def current_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict:
        raise NotImplementedError
