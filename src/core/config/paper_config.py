from dataclasses import dataclass

from core.config.live_config import LiveConfig


@dataclass
class PaperConfig(LiveConfig):
    """Live + paper-брокер в SimulatedExchange: комиссия/slippage как в бэктесте (TAKER_COM, SLIPPAGE)."""

    initial_capital: float = 100.0
    taker_com: float = 0.0004
    slippage: float = 0.0003

    @classmethod
    def from_env(cls) -> "PaperConfig":
        cfg = LiveConfig.from_env()
        return cls(
            symbols=cfg.symbols,
            model_path=cfg.model_path,
            timeframe=cfg.timeframe,
            htf_timeframe=cfg.htf_timeframe,
            api_key=cfg.api_key,
            secret=cfg.secret,
            testnet=True,
            ws_url=cfg.ws_url,
            leverage=cfg.leverage,
            risk_per_trade=cfg.risk_per_trade,
            allow_longs=cfg.allow_longs,
            allow_shorts=cfg.allow_shorts,
            directional_proba_threshold=cfg.directional_proba_threshold,
            min_signal_gap=cfg.min_signal_gap,
            taker_com=float(cls.env_str("TAKER_COM", "0.0004")),
            slippage=float(cls.env_str("SLIPPAGE", "0.0003")),
            initial_capital=float(cls.env_str("PAPER_CAPITAL", "100.0")),
        )
