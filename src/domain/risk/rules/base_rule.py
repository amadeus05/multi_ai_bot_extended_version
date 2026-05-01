from abc import ABC, abstractmethod

from domain.risk.models.risk_context import RiskContext


class RiskRule(ABC):
    @abstractmethod
    def apply(self, ctx: RiskContext) -> bool:
        raise NotImplementedError
