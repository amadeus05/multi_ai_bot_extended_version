from abc import ABC, abstractmethod
from typing import Optional

from core.interfaces.exchange import Exchange
from core.types.domain_types import Order


class RiskManager(ABC):
    @abstractmethod
    def check(self, order: Order, portfolio: "PortfolioManager", exchange: Exchange) -> Optional[Order]:
        raise NotImplementedError
