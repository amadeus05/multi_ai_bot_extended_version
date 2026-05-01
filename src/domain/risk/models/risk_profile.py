from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    leverage: float = 1.0
    risk_per_trade: float = 0.1
    max_new_positions_per_bar: int = 1
    max_open_positions: int = 1
    sl_cooldown_bars: int = 8
    max_sl_per_day: int = 3
    reduce_risk_after_consecutive_losses: int = 2
    reduced_risk_per_trade: float = 0.005
