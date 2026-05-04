from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalSettings:
    allow_longs: bool = True
    allow_shorts: bool = True
    directional_proba_threshold: float = 0.55
    min_signal_gap: float = 0.05


@dataclass(frozen=True)
class RiskSettings:
    risk_per_trade: float = 0.1
    max_new_positions_per_bar: int = 1
    max_open_positions: int = 1
    sl_cooldown_bars: int = 8
    max_sl_per_day: int = 3
    reduce_risk_after_consecutive_losses: int = 2
    reduced_risk_per_trade: float = 0.005
    min_position_notional: float = 10.0


@dataclass(frozen=True)
class ExecutionCostSettings:
    taker_com: float = 0.0004
    slippage: float = 0.0003


@dataclass(frozen=True)
class NotificationSettings:
    channels: tuple[str, ...] = ("silent",)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


@dataclass(frozen=True)
class TradingSettings:
    symbols: tuple[str, ...] = ("BTC/USDT",)
    model_path: str = "models/latest.pkl"
    timeframe: str = "1h"
    htf_timeframe: str = "4h"
    initial_capital: float = 10000.0
    leverage: float = 1.0
    signal: SignalSettings = field(default_factory=SignalSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    costs: ExecutionCostSettings = field(default_factory=ExecutionCostSettings)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)


@dataclass(frozen=True)
class BacktestSettings:
    trading: TradingSettings
    dataset_dir: str = "data/raw/labeled/source=bybit"
    charts_dir: str = "backtest_charts"
    skip_initial_bars: int = 0
    start: str = ""
    end: str = ""
    strict_oos: bool = False

    @property
    def symbols(self) -> list[str]:
        return list(self.trading.symbols)

    @property
    def timeframe(self) -> str:
        return self.trading.timeframe

    @property
    def htf_timeframe(self) -> str:
        return self.trading.htf_timeframe

    @property
    def initial_capital(self) -> float:
        return self.trading.initial_capital

    @property
    def leverage(self) -> float:
        return self.trading.leverage

    @property
    def backtest_charts_dir(self) -> str:
        return self.charts_dir


@dataclass(frozen=True)
class PaperSettings:
    trading: TradingSettings
    ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    testnet: bool = True
    exit_heartbeat_enabled: bool = True
    exit_timeframe: str = "1m"

    @property
    def symbols(self) -> list[str]:
        return list(self.trading.symbols)


@dataclass(frozen=True)
class LiveSettings:
    trading: TradingSettings
    api_key: str = ""
    api_secret: str = ""
    ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    testnet: bool = False

    @property
    def symbols(self) -> list[str]:
        return list(self.trading.symbols)
