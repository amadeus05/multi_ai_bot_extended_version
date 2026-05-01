from dataclasses import dataclass

from core.config.base import BaseConfig


@dataclass
class LiveConfig(BaseConfig):
    api_key: str
    secret: str
    testnet: bool
    ws_url: str
    leverage: float
    risk_per_trade: float
    allow_longs: bool
    allow_shorts: bool
    directional_proba_threshold: float
    min_signal_gap: float

    @classmethod
    def from_env(cls) -> "LiveConfig":
        return cls(
            symbols=cls.env_list("SYMBOLS", "BTC/USDT"),
            model_path=cls.env_str("MODEL_PATH", "models/latest.pkl"),
            timeframe=cls.env_str("TIMEFRAME", "1h"),
            htf_timeframe=cls.env_str("HTF_TIMEFRAME", "4h"),
            api_key=cls.env_str("API_KEY"),
            secret=cls.env_str("API_SECRET"),
            testnet=False,
            ws_url=cls.env_str("WS_URL", "wss://stream.bybit.com/v5/public/linear"),
            leverage=float(cls.env_str("LEVERAGE", "1")),
            risk_per_trade=float(cls.env_str("RISK_PER_TRADE", "0.1")),
            allow_longs=cls.env_str("ALLOW_LONGS", "1").strip().lower() in ("1", "true", "yes"),
            allow_shorts=cls.env_str("ALLOW_SHORTS", "1").strip().lower() in ("1", "true", "yes"),
            directional_proba_threshold=float(cls.env_str("DIRECTIONAL_PROBA_THRESHOLD", "0.5")),
            min_signal_gap=float(cls.env_str("MIN_SIGNAL_GAP", "0.0")),
        )
