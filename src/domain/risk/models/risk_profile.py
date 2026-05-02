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
    min_position_notional: float = 10.0

    @classmethod
    def from_settings(cls, settings, *, leverage: float) -> "RiskProfile":
        return cls(
            leverage=float(leverage),
            risk_per_trade=float(settings.risk_per_trade),
            max_new_positions_per_bar=int(settings.max_new_positions_per_bar),
            max_open_positions=int(settings.max_open_positions),
            sl_cooldown_bars=int(settings.sl_cooldown_bars),
            max_sl_per_day=int(settings.max_sl_per_day),
            reduce_risk_after_consecutive_losses=int(settings.reduce_risk_after_consecutive_losses),
            reduced_risk_per_trade=float(settings.reduced_risk_per_trade),
            min_position_notional=float(settings.min_position_notional),
        )
