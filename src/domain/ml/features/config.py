from __future__ import annotations

import json
from typing import Any

from core.config.base import BaseConfig

# Соответствует легаси-прогону (old_results.txt): 21 фич после отключения 77 лишних,
# один символ BTC/USDT, clip 1%/99%, labeling hybrid_v1 / train v2.
LEGACY_MVP_V1_FEATURES: list[str] = [
    "adx_4h",
    "atr_ratio_1h",
    "cross_sectional_rank_ema_fast_slow_1h",
    "distance_to_resistance_1h",
    "distance_to_support_1h",
    "ema_fast_slow",
    "ema_slope_4h",
    "flat_efficiency_1h_24",
    "funding_rate_8h",
    "market_breadth_ema_fast_slow_1h",
    "market_breadth_pos_return_4h_3",
    "mcc_sign_agreement_btc_24h",
    "premium_index_change_24h",
    "price_position_1h",
    "price_position_4h",
    "range_center_distance_atr_1h_48",
    "realized_vol_4h_returns_20",
    "return_4h_14",
    "rsi_1h",
    "volatility_regime_change_1h",
    "zscore_vs_vwap_4h",
]

FEATURE_PROFILES: dict[str, Any] = {
    # Полный набор (98+ имён) — для мульти-экспериментов; по умолчанию не используется.
    "all": "__all__",
    "empty": [],
    # Устаревший тестовый набор — закомментирован в пользу legacy_mvp_v1.
    # "base_only": [
    #     "realized_vol_1h",
    #     "ema_fast_slow",
    #     "return_1h_24",
    #     "linear_regression_slope_atr_1h_24",
    #     "trend_efficiency_24h",
    #     "atr_ratio_1h",
    #     "range_compression_1h",
    #     "price_position_1h",
    #     "relative_strength_vs_btc_24h",
    #     "residual_return_24h",
    #     "return_4h_3",
    #     "return_4h_14",
    #     "ema_slope_4h",
    #     "realized_vol_4h_returns_20",
    #     "market_breadth_pos_return_4h_3",
    #     "market_dispersion_return_4h_3",
    #     "zscore_vs_vwap_4h",
    #     "vol_ratio",
    #     "price_position_4h",
    #     "adx_4h",
    # ],
    "legacy_mvp_v1": LEGACY_MVP_V1_FEATURES,
}

FEATURE_BUILD_REQUEST: dict[str, Any] = {
    "profile": "legacy_mvp_v1",
    # Технические зависимости для построения части market-context фич.
    # В train не попадут, т.к. TRAIN_FEATURE_SUBSET=legacy_mvp_v1 оставляет ровно 21 фичу.
    "include_features": ["return_4h_3", "realized_vol_1h"],
    "exclude_features": [],
    "exclude_blocks": [],
}


def _env_int(name: str, default: int) -> int:
    return int(BaseConfig.env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(BaseConfig.env_str(name, str(default)))


def _env_json(name: str, default: Any) -> Any:
    raw = BaseConfig.env_str(name, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def __getattr__(name: str) -> Any:
    if name == "FEATURE_PROFILES":
        return _env_json("FEATURE_PROFILES_JSON", FEATURE_PROFILES)
    if name == "FEATURE_BUILD_REQUEST":
        return _env_json("FEATURE_BUILD_REQUEST_JSON", FEATURE_BUILD_REQUEST)

    env_defaults: dict[str, Any] = {
        "EMA_FAST_WINDOW": 12,
        "EMA_SLOW_WINDOW": 48,
        "REALIZED_VOL_WINDOW_1H": 24,
        "RANGE_COMPRESSION_SHORT_WINDOW_1H": 12,
        "RANGE_COMPRESSION_LONG_WINDOW_1H": 48,
        "VOLUME_RATIO_WINDOW_1H": 24,
        "VOLUME_ZSCORE_WINDOW_1H": 24 * 7,
        "VOLUME_24H_WINDOW_1H": 24,
        "REALIZED_VOL_WINDOW_4H": 20,
        "RANGE_WINDOW_4H": 14,
        "VWAP_WINDOW_4H": 20,
        "EMA_SLOPE_BASE_WINDOW_4H": 21,
        "EMA_SLOPE_WINDOW_4H": 6,
        "MARKET_ZSCORE_WINDOW": 96,
        "OPEN_INTEREST_ZSCORE_WINDOW_1H": 24 * 7,
        "OPEN_INTEREST_CHANGE_LOOKBACK_8H": 8,
        "OPEN_INTEREST_CHANGE_LOOKBACK_24H": 24,
        "PREMIUM_INDEX_ZSCORE_WINDOW_1H": 24 * 7,
        "PREMIUM_INDEX_CHANGE_LOOKBACK_1H": 24,
        "FUNDING_ZSCORE_WINDOW_1H": 24 * 7,
        "FUNDING_CHANGE_LOOKBACK_1H": 24,
    }
    if name in env_defaults:
        default = env_defaults[name]
        if isinstance(default, float):
            return _env_float(name, default)
        return _env_int(name, int(default))
    raise AttributeError(name)
