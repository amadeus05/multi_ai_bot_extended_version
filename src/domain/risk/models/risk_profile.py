import os
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
    def from_env(cls) -> "RiskProfile":
        """Те же переменные окружения, что и BacktestConfig для торговых лимитов."""
        return cls(
            leverage=float(os.getenv("LEVERAGE", "1")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.1")),
            max_new_positions_per_bar=int(os.getenv("BACKTEST_MAX_NEW_POSITIONS_PER_BAR", "1")),
            max_open_positions=int(os.getenv("BACKTEST_MAX_OPEN_POSITIONS", "1")),
            sl_cooldown_bars=int(os.getenv("BACKTEST_SL_COOLDOWN_BARS", "8")),
            max_sl_per_day=int(os.getenv("BACKTEST_MAX_SL_PER_DAY", "3")),
            reduce_risk_after_consecutive_losses=int(
                os.getenv("BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", "2")
            ),
            reduced_risk_per_trade=float(os.getenv("BACKTEST_REDUCED_RISK_PER_TRADE", "0.005")),
            min_position_notional=float(os.getenv("BACKTEST_MIN_POSITION_NOTIONAL", "10")),
        )
