from abc import ABC, abstractmethod
from typing import Awaitable, Callable

import pandas as pd

from core.types.domain_types import Tick


class DataProvider(ABC):
    @abstractmethod
    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        raise NotImplementedError

    def subscribe_exit_checks(self, symbol: str, callback: Callable[[Tick], Awaitable[None]]) -> None:
        """Optional faster market-data stream for TP/SL checks."""
        return None

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError
