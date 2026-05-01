from dataclasses import dataclass

from core.types.domain_types import Order
from domain.risk.models.risk_profile import RiskProfile


@dataclass
class RiskContext:
    order: Order
    portfolio: object
    exchange: object
    profile: RiskProfile
    quote_asset: str
    bar_index: int
    current_bar_ts: object
    sl_per_day: dict[str, int]
    symbol_cooldown_until_bar: dict[str, int]
    new_positions_in_bar: int
    consecutive_losses: int
