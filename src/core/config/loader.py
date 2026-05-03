from __future__ import annotations

import os
from dataclasses import fields, is_dataclass, replace
from typing import Any

from core.config.base import BaseConfig
from core.config.settings import (
    BacktestSettings,
    ExecutionCostSettings,
    LiveSettings,
    PaperSettings,
    RiskSettings,
    SignalSettings,
    TradingSettings,
)
from core.config.storage_config import StorageSettings


def _load_env() -> None:
    BaseConfig._load_dotenv()


def _env(names: str | tuple[str, ...], default: str) -> str:
    _load_env()
    keys = (names,) if isinstance(names, str) else names
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def _bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _symbols(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or ("BTC/USDT",)


def _apply_overrides(settings: Any, overrides: dict[str, Any] | None) -> Any:
    out = settings
    for path, value in (overrides or {}).items():
        out = _replace_path(out, str(path).split("."), value)
    return out


def _replace_path(obj: Any, parts: list[str], value: Any) -> Any:
    if not parts:
        return value
    if not is_dataclass(obj):
        raise KeyError(f"Cannot override {'.'.join(parts)!r}: target is not a dataclass")
    name = parts[0]
    allowed = {field.name for field in fields(obj)}
    if name not in allowed:
        raise KeyError(f"Unknown settings override field: {name!r}")
    current = getattr(obj, name)
    return replace(obj, **{name: _replace_path(current, parts[1:], value)})


def load_trading_settings(
    *,
    mode: str,
    overrides: dict[str, Any] | None = None,
) -> TradingSettings:
    mode_key = mode.strip().lower()
    default_capital = {
        "backtest": "10000",
        "paper": "100.0",
        "live": "5000.0",
    }.get(mode_key, "10000")
    capital_keys = {
        "backtest": ("TRADING_INITIAL_CAPITAL", "INITIAL_CAPITAL"),
        "paper": ("PAPER_CAPITAL", "TRADING_INITIAL_CAPITAL", "INITIAL_CAPITAL"),
        "live": ("LIVE_INITIAL_CAPITAL", "TRADING_INITIAL_CAPITAL", "INITIAL_CAPITAL"),
    }.get(mode_key, ("TRADING_INITIAL_CAPITAL", "INITIAL_CAPITAL"))

    settings = TradingSettings(
        symbols=_symbols(_env(("TRADING_SYMBOLS", "SYMBOLS"), "BTC/USDT")),
        model_path=_env(("TRADING_MODEL_PATH", "MODEL_PATH"), "models/latest.pkl"),
        timeframe=_env(("TRADING_TIMEFRAME", "TIMEFRAME"), "1h"),
        htf_timeframe=_env(("TRADING_HTF_TIMEFRAME", "HTF_TIMEFRAME"), "4h"),
        initial_capital=float(_env(capital_keys, default_capital)),
        leverage=float(_env(("TRADING_LEVERAGE", "LEVERAGE"), "1")),
        signal=SignalSettings(
            allow_longs=_bool(_env(("SIGNAL_ALLOW_LONGS", "ALLOW_LONGS"), "1")),
            allow_shorts=_bool(_env(("SIGNAL_ALLOW_SHORTS", "ALLOW_SHORTS"), "1")),
            directional_proba_threshold=float(
                _env(("SIGNAL_PROBA_THRESHOLD", "DIRECTIONAL_PROBA_THRESHOLD"), "0.55")
            ),
            min_signal_gap=float(_env(("SIGNAL_MIN_GAP", "MIN_SIGNAL_GAP"), "0.05")),
        ),
        risk=RiskSettings(
            risk_per_trade=float(_env(("RISK_PER_TRADE",), "0.1")),
            max_new_positions_per_bar=int(
                _env(("RISK_MAX_NEW_POSITIONS_PER_BAR", "BACKTEST_MAX_NEW_POSITIONS_PER_BAR"), "1")
            ),
            max_open_positions=int(_env(("RISK_MAX_OPEN_POSITIONS", "BACKTEST_MAX_OPEN_POSITIONS"), "1")),
            sl_cooldown_bars=int(_env(("RISK_SL_COOLDOWN_BARS", "BACKTEST_SL_COOLDOWN_BARS"), "8")),
            max_sl_per_day=int(_env(("RISK_MAX_SL_PER_DAY", "BACKTEST_MAX_SL_PER_DAY"), "3")),
            reduce_risk_after_consecutive_losses=int(
                _env(
                    (
                        "RISK_REDUCE_AFTER_CONSECUTIVE_LOSSES",
                        "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES",
                    ),
                    "2",
                )
            ),
            reduced_risk_per_trade=float(
                _env(("RISK_REDUCED_RISK_PER_TRADE", "BACKTEST_REDUCED_RISK_PER_TRADE"), "0.005")
            ),
            min_position_notional=float(
                _env(("RISK_MIN_POSITION_NOTIONAL", "BACKTEST_MIN_POSITION_NOTIONAL"), "10")
            ),
        ),
        costs=ExecutionCostSettings(
            taker_com=float(_env(("EXEC_TAKER_COM", "TAKER_COM"), "0.0004")),
            slippage=float(_env(("EXEC_SLIPPAGE", "SLIPPAGE"), "0.0003")),
        ),
    )
    return _apply_overrides(settings, overrides)


def load_backtest_settings(overrides: dict[str, Any] | None = None) -> BacktestSettings:
    settings = BacktestSettings(
        trading=load_trading_settings(mode="backtest"),
        dataset_dir=_env(("BACKTEST_DATASET_DIR", "DATASET_DIR"), "data/raw/labeled/source=bybit"),
        charts_dir=_env("BACKTEST_CHARTS_DIR", "backtest_charts"),
        skip_initial_bars=int(_env("BACKTEST_SKIP_INITIAL_BARS", "0")),
        start=_env("BACKTEST_START", "").strip(),
        end=_env("BACKTEST_END", "").strip(),
        strict_oos=_bool(_env("BACKTEST_STRICT_OOS", "0")),
    )
    return _apply_overrides(settings, overrides)


def load_paper_settings(overrides: dict[str, Any] | None = None) -> PaperSettings:
    settings = PaperSettings(
        trading=load_trading_settings(mode="paper"),
        ws_url=_env("WS_URL", "wss://stream.bybit.com/v5/public/linear"),
        testnet=True,
    )
    return _apply_overrides(settings, overrides)


def load_live_settings(overrides: dict[str, Any] | None = None) -> LiveSettings:
    settings = LiveSettings(
        trading=load_trading_settings(mode="live"),
        api_key=_env("API_KEY", ""),
        api_secret=_env("API_SECRET", ""),
        ws_url=_env("WS_URL", "wss://stream.bybit.com/v5/public/linear"),
        testnet=_bool(_env("LIVE_TESTNET", "0")),
    )
    return _apply_overrides(settings, overrides)


def load_storage_settings(overrides: dict[str, Any] | None = None) -> StorageSettings:
    settings = StorageSettings(
        driver=_env("STORAGE_DRIVER", "sqlite").strip().lower(),
        sqlite_path=_env(("STORAGE_SQLITE_PATH", "SQLITE_PATH"), "data/trading_runtime.sqlite"),
        supabase_url=_env("SUPABASE_URL", ""),
        supabase_service_key=_env(("SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY"), ""),
        supabase_schema=_env("SUPABASE_SCHEMA", "public"),
        events_table=_env("STORAGE_EVENTS_TABLE", "trading_events"),
        journal_enabled=_bool(_env("EVENT_JOURNAL_ENABLED", "0")),
        journal_session_id=_env("EVENT_JOURNAL_SESSION_ID", ""),
        journal_source=_env("EVENT_JOURNAL_SOURCE", "trading_runtime"),
    )
    return _apply_overrides(settings, overrides)
