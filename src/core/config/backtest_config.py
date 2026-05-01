from dataclasses import dataclass

from core.config.base import BaseConfig


@dataclass
class BacktestConfig(BaseConfig):
    dataset_dir: str
    commission: float
    initial_capital: float
    leverage: float
    risk_per_trade: float
    backtest_max_new_positions_per_bar: int
    backtest_max_open_positions: int
    backtest_sl_cooldown_bars: int
    backtest_max_sl_per_day: int
    backtest_reduce_risk_after_consecutive_losses: int
    backtest_reduced_risk_per_trade: float
    split_mode: str
    monthly_train_months: int
    monthly_test_months: int
    monthly_window_mode: str
    allow_longs: bool
    allow_shorts: bool
    directional_proba_threshold: float
    min_signal_gap: float
    taker_com: float
    slippage: float
    min_position_notional: float
    backtest_charts_dir: str
    backtest_skip_initial_bars: int
    backtest_start: str
    backtest_end: str
    backtest_strict_oos: bool

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        return cls(
            symbols=cls.env_list("SYMBOLS", "BTC/USDT"),
            model_path=cls.env_str("MODEL_PATH", "models/latest.pkl"),
            timeframe=cls.env_str("TIMEFRAME", "1h"),
            htf_timeframe=cls.env_str("HTF_TIMEFRAME", "4h"),
            dataset_dir=cls.env_str("DATASET_DIR", "data/raw/labeled/source=bybit"),
            commission=float(cls.env_str("COMMISSION", "0.0005")),
            initial_capital=float(cls.env_str("INITIAL_CAPITAL", "10000")),
            leverage=float(cls.env_str("LEVERAGE", "1")),
            risk_per_trade=float(cls.env_str("RISK_PER_TRADE", "0.1")),
            backtest_max_new_positions_per_bar=int(cls.env_str("BACKTEST_MAX_NEW_POSITIONS_PER_BAR", "1")),
            backtest_max_open_positions=int(cls.env_str("BACKTEST_MAX_OPEN_POSITIONS", "1")),
            backtest_sl_cooldown_bars=int(cls.env_str("BACKTEST_SL_COOLDOWN_BARS", "8")),
            backtest_max_sl_per_day=int(cls.env_str("BACKTEST_MAX_SL_PER_DAY", "3")),
            backtest_reduce_risk_after_consecutive_losses=int(
                cls.env_str("BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", "2")
            ),
            backtest_reduced_risk_per_trade=float(cls.env_str("BACKTEST_REDUCED_RISK_PER_TRADE", "0.005")),
            split_mode=cls.env_str("BACKTEST_SPLIT_MODE", "monthly"),
            monthly_train_months=int(cls.env_str("BACKTEST_MONTHLY_TRAIN_MONTHS", "6")),
            monthly_test_months=int(cls.env_str("BACKTEST_MONTHLY_TEST_MONTHS", "1")),
            monthly_window_mode=cls.env_str("BACKTEST_MONTHLY_WINDOW_MODE", "expanding"),
            allow_longs=cls.env_str("ALLOW_LONGS", "1").strip().lower() in ("1", "true", "yes"),
            allow_shorts=cls.env_str("ALLOW_SHORTS", "1").strip().lower() in ("1", "true", "yes"),
            directional_proba_threshold=float(cls.env_str("DIRECTIONAL_PROBA_THRESHOLD", "0.5")),
            min_signal_gap=float(cls.env_str("MIN_SIGNAL_GAP", "0.0")),
            taker_com=float(cls.env_str("TAKER_COM", "0.0004")),
            slippage=float(cls.env_str("SLIPPAGE", "0.0003")),
            min_position_notional=float(cls.env_str("BACKTEST_MIN_POSITION_NOTIONAL", "10")),
            backtest_charts_dir=cls.env_str("BACKTEST_CHARTS_DIR", "backtest_charts"),
            backtest_skip_initial_bars=int(cls.env_str("BACKTEST_SKIP_INITIAL_BARS", "0")),
            backtest_start=cls.env_str("BACKTEST_START", "").strip(),
            backtest_end=cls.env_str("BACKTEST_END", "").strip(),
            backtest_strict_oos=cls.env_str("BACKTEST_STRICT_OOS", "0").strip().lower()
            in ("1", "true", "yes"),
        )
