from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AdaptiveHorizonConfig:
    enabled: bool = True
    base_horizon: int = 12
    min_horizon: int = 8
    max_horizon: int = 20
    vol_low: float = 0.005
    vol_high: float = 0.025


@dataclass(frozen=True)
class BarrierConfig:
    use_dynamic_barriers: bool = True
    stop_pct: float = 0.015
    take_pct: float = 0.03
    atr_multiplier: float = 1.25
    rvol_multiplier: float = 0.75
    tp_to_sl_ratio: float = 2.0
    min_pct: float = 0.0075
    max_pct: float = 0.06
    taker_com: float = 0.0004
    slippage: float = 0.0003


@dataclass(frozen=True)
class LabelingConfig:
    adaptive_horizon: AdaptiveHorizonConfig = AdaptiveHorizonConfig()
    barrier: BarrierConfig = BarrierConfig()
    realized_vol_column: str = "realized_vol_1h"

    @classmethod
    def from_env(cls) -> "LabelingConfig":
        return cls(
            adaptive_horizon=AdaptiveHorizonConfig(
                enabled=os.getenv("ENABLE_ADAPTIVE_HORIZON", "1") == "1",
                base_horizon=int(os.getenv("HORIZON", "12")),
                min_horizon=int(os.getenv("ADAPTIVE_HORIZON_MIN", "8")),
                max_horizon=int(os.getenv("ADAPTIVE_HORIZON_MAX", "20")),
                vol_low=float(os.getenv("ADAPTIVE_HORIZON_VOL_LOW", "0.005")),
                vol_high=float(os.getenv("ADAPTIVE_HORIZON_VOL_HIGH", "0.025")),
            ),
            barrier=BarrierConfig(
                use_dynamic_barriers=os.getenv("USE_DYNAMIC_BARRIERS", "1") == "1",
                stop_pct=float(os.getenv("SL_PCT", "0.015")),
                take_pct=float(os.getenv("TP_PCT", "0.03")),
                atr_multiplier=float(os.getenv("BARRIER_ATR_MULTIPLIER", "1.25")),
                rvol_multiplier=float(os.getenv("BARRIER_RVOL_MULTIPLIER", "0.75")),
                tp_to_sl_ratio=float(os.getenv("BARRIER_TP_TO_SL_RATIO", "2.0")),
                min_pct=float(os.getenv("BARRIER_MIN_PCT", "0.0075")),
                max_pct=float(os.getenv("BARRIER_MAX_PCT", "0.06")),
                taker_com=float(os.getenv("EXEC_TAKER_COM", os.getenv("TAKER_COM", "0.0004"))),
                slippage=float(os.getenv("EXEC_SLIPPAGE", os.getenv("SLIPPAGE", "0.0003"))),
            ),
            realized_vol_column=os.getenv("REALIZED_VOL_COLUMN", "realized_vol_1h"),
        )
