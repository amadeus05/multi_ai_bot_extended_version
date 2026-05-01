from abc import ABC, abstractmethod
from typing import Optional

from core.types.domain_types import Order, Tick


class Strategy(ABC):
    @abstractmethod
    def on_prediction(self, tick: Tick, prediction: dict, portfolio: "PortfolioManager") -> Optional[Order]:
        raise NotImplementedError
