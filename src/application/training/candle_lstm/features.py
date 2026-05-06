from __future__ import annotations

import numpy as np
import pandas as pd

from domain.ml.features.indicators import compute_atr


CANDLE_LSTM_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_vs_ema_20",
    "close_vs_ema_50",
    "ema20_slope_3",
    "atr14_norm",
    "realized_vol_24",
    "volume_norm_24",
    "funding_rate",
    "funding_rate_change_8h",
    "premium_index_close",
    "premium_index_change_8h",
    "open_interest_change_8h",
    "btc_return_1h",
    "asset_minus_btc_return_1h",
]


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator.astype(float) / denominator.astype(float)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio > 0)
    return np.log(ratio)


def pct_delta(series: pd.Series, periods: int) -> pd.Series:
    base = series.shift(periods).replace(0, np.nan)
    return ((series - base) / base).replace([np.inf, -np.inf], np.nan)


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_candle_lstm_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol"] + CANDLE_LSTM_FEATURE_COLUMNS)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candle LSTM feature frame is missing columns: {missing}")

    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source = source.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for symbol, symbol_frame in source.groupby("symbol", observed=True):
        frames.append(_build_symbol_candle_lstm_features(symbol_frame, str(symbol)))
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol"] + CANDLE_LSTM_FEATURE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _build_symbol_candle_lstm_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    close = _numeric_column(frame, "close")
    open_ = _numeric_column(frame, "open")
    high = _numeric_column(frame, "high")
    low = _numeric_column(frame, "low")
    volume = _numeric_column(frame, "volume")
    funding_rate = _numeric_column(frame, "funding_rate").fillna(0.0)
    premium_index = _numeric_column(frame, "premium_index_close").fillna(0.0)
    open_interest = _numeric_column(frame, "open_interest")

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    atr14 = compute_atr(high, low, close, length=14)
    log_return_1 = safe_log_ratio(close, close.shift(1))

    output = pd.DataFrame({"timestamp": frame["timestamp"].copy()})
    output["symbol"] = symbol
    output["log_return_1"] = log_return_1
    output["log_return_4"] = safe_log_ratio(close, close.shift(4))
    output["body_pct"] = ((close - open_) / open_).replace([np.inf, -np.inf], np.nan)
    output["range_pct"] = ((high - low) / open_).replace([np.inf, -np.inf], np.nan)
    output["upper_wick_pct"] = ((high - np.maximum(open_, close)) / open_).replace([np.inf, -np.inf], np.nan)
    output["lower_wick_pct"] = ((np.minimum(open_, close) - low) / open_).replace([np.inf, -np.inf], np.nan)
    output["close_vs_ema_20"] = ((close - ema20) / close).replace([np.inf, -np.inf], np.nan)
    output["close_vs_ema_50"] = ((close - ema50) / close).replace([np.inf, -np.inf], np.nan)
    output["ema20_slope_3"] = pct_delta(ema20, 3)
    output["atr14_norm"] = (atr14 / close).replace([np.inf, -np.inf], np.nan)
    output["realized_vol_24"] = log_return_1.rolling(24).std()
    volume_mean_24 = volume.rolling(24).mean().replace(0, np.nan)
    output["volume_norm_24"] = safe_log_ratio(volume, volume_mean_24)
    output["funding_rate"] = funding_rate
    output["funding_rate_change_8h"] = (funding_rate - funding_rate.shift(8)).fillna(0.0)
    output["premium_index_close"] = premium_index
    output["premium_index_change_8h"] = (premium_index - premium_index.shift(8)).fillna(0.0)
    output["open_interest_change_8h"] = pct_delta(open_interest, 8).fillna(0.0)
    output["btc_return_1h"] = _resolve_btc_return(frame, symbol, log_return_1)
    output["asset_minus_btc_return_1h"] = (output["log_return_1"] - output["btc_return_1h"]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    return output[["timestamp", "symbol"] + CANDLE_LSTM_FEATURE_COLUMNS].copy()


def _resolve_btc_return(frame: pd.DataFrame, symbol: str, log_return_1: pd.Series) -> pd.Series:
    if "btc_return_1h" in frame.columns:
        return pd.to_numeric(frame["btc_return_1h"], errors="coerce").fillna(0.0)
    if symbol == "BTC/USDT":
        return log_return_1.fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)
