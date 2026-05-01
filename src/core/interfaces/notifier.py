from abc import ABC, abstractmethod

from core.types.domain_types import Order, Trade


class Notifier(ABC):
    @abstractmethod
    async def alert(self, message: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def log_trade(self, trade: Trade) -> None:
        raise NotImplementedError

    @abstractmethod
    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        raise NotImplementedError
