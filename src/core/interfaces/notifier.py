from abc import ABC, abstractmethod

from core.types.domain_types import Order
from core.types.notifications import SignalNotification, SystemNotification, TradeExitNotification


class Notifier(ABC):
    @property
    def wants_signal_charts(self) -> bool:
        return False

    @abstractmethod
    async def notify_signal(self, notification: SignalNotification) -> None:
        raise NotImplementedError

    @abstractmethod
    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        raise NotImplementedError

    @abstractmethod
    async def notify_system(self, notification: SystemNotification) -> None:
        raise NotImplementedError

    @abstractmethod
    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        raise NotImplementedError
