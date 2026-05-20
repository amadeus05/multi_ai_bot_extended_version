from __future__ import annotations

import numpy as np
import pandas as pd

from domain.ml.labeling.models import LabelingConfig

BASE_OUTPUT_COLUMNS = ["timestamp", "decision_time", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def compute_effective_horizons(df: pd.DataFrame, cfg: LabelingConfig) -> np.ndarray:
    base_horizon = max(1, int(cfg.adaptive_horizon.base_horizon))
    if not cfg.adaptive_horizon.enabled:
        return np.full(len(df), base_horizon, dtype=np.int32)

    realized_vol_col = cfg.realized_vol_column
    if realized_vol_col not in df.columns:
        return np.full(len(df), base_horizon, dtype=np.int32)

    min_horizon = int(cfg.adaptive_horizon.min_horizon)
    max_horizon = int(cfg.adaptive_horizon.max_horizon)
    if min_horizon > max_horizon:
        min_horizon, max_horizon = max_horizon, min_horizon
    min_horizon = max(1, min_horizon)
    max_horizon = max(min_horizon, max_horizon)

    vol_low = float(cfg.adaptive_horizon.vol_low)
    vol_high = float(cfg.adaptive_horizon.vol_high)
    if not np.isfinite(vol_low) or not np.isfinite(vol_high) or vol_high <= vol_low:
        return np.full(len(df), base_horizon, dtype=np.int32)

    vol = pd.Series(df[realized_vol_col], copy=False).astype(float).abs()
    normalized = ((vol - vol_low) / (vol_high - vol_low)).clip(lower=0.0, upper=1.0)
    normalized_values = normalized.to_numpy()
    adaptive_raw = np.rint(max_horizon - normalized_values * (max_horizon - min_horizon))
    adaptive = np.full(len(df), base_horizon, dtype=np.int32)
    valid_mask = np.isfinite(adaptive_raw)
    adaptive[valid_mask] = np.clip(adaptive_raw[valid_mask], min_horizon, max_horizon).astype(np.int32)
    return adaptive


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
    realized_vol_1h: pd.Series,
    cfg: LabelingConfig,
    effective_horizons: np.ndarray | None = None,
) -> pd.Series:
    atr_pct = safe_ratio(atr_14, close).abs()
    if effective_horizons is None:
        horizon_sqrt = np.sqrt(float(max(1, cfg.adaptive_horizon.base_horizon)))
    else:
        horizon_sqrt = np.sqrt(np.maximum(effective_horizons.astype(float), 1.0))
    horizon_vol_pct = realized_vol_1h.abs() * horizon_sqrt

    stop_pct = pd.concat(
        [
            atr_pct * float(cfg.barrier.atr_multiplier),
            horizon_vol_pct * float(cfg.barrier.rvol_multiplier),
        ],
        axis=1,
    ).max(axis=1)

    min_pct = float(cfg.barrier.min_pct)
    max_pct = float(cfg.barrier.max_pct)
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series, cfg: LabelingConfig) -> pd.Series:
    return stop_pct * float(cfg.barrier.tp_to_sl_ratio)


def attach_barrier_columns(df: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)
    effective_horizons = compute_effective_horizons(output, cfg)

    realized_vol_col = cfg.realized_vol_column
    if cfg.barrier.use_dynamic_barriers:
        if realized_vol_col not in output.columns:
            raise ValueError("Dynamic barriers require feature 'realized_vol_1h' to be enabled.")
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(
            close,
            atr_14,
            output[realized_vol_col],
            cfg,
            effective_horizons=effective_horizons,
        )
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"], cfg)
    else:
        output["barrier_stop_pct"] = float(cfg.barrier.stop_pct)
        output["barrier_take_pct"] = float(cfg.barrier.take_pct)
    return output


def compute_clean_pnl(direction: int, entry_price: float, exit_price: float, cfg: LabelingConfig) -> float:
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    taker_com = float(cfg.barrier.taker_com)
    return raw_pnl - (taker_com + taker_com)


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
    *,
    slippage: float,
) -> tuple[float | None, str | None]:
    slippage = float(slippage)
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)
        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - slippage)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - slippage)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)
        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + slippage)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + slippage)
            return exit_price, "TP"
    return None, None


def simulate_trade_outcome(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    stop_pcts: np.ndarray,
    take_pcts: np.ndarray,
    start_idx: int,
    direction: int,
    horizon: int,
    cfg: LabelingConfig,
) -> tuple[float, str | None]:
    slippage = float(cfg.barrier.slippage)
    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 + slippage) if direction == 1 else base_open * (1 - slippage)
    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if np.isnan(stop_pct) or np.isnan(take_pct):
        return 0.0, None

    for j in range(1, horizon + 1):
        candle_idx = start_idx + j
        if candle_idx >= len(opens):
            break
        exit_price, reason = resolve_trade_exit(
            direction,
            entry_price,
            opens[candle_idx],
            highs[candle_idx],
            lows[candle_idx],
            stop_pct,
            take_pct,
            slippage=slippage,
        )
        if exit_price is not None:
            return compute_clean_pnl(direction, entry_price, exit_price, cfg), reason
    return 0.0, None


def triple_barrier_labeling(df: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    labels = []
    effective_horizons = compute_effective_horizons(df, cfg)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else max(1, cfg.adaptive_horizon.base_horizon)

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - max_horizon):
        label = 0
        horizon = int(effective_horizons[i]) if i < len(effective_horizons) else max(1, cfg.adaptive_horizon.base_horizon)
        long_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=1, horizon=horizon, cfg=cfg)
        short_pnl, _ = simulate_trade_outcome(opens, highs, lows, stop_pcts, take_pcts, i, direction=-1, horizon=horizon, cfg=cfg)

        if long_pnl > 0 and short_pnl <= 0:
            label = 1
        elif short_pnl > 0 and long_pnl <= 0:
            label = -1
        labels.append(label)

    labels.extend([0] * max_horizon)
    output = df.copy()
    output["Target"] = labels
    return output


def finalize_feature_frame(df: pd.DataFrame, feature_columns: list[str], cfg: LabelingConfig) -> pd.DataFrame:
    output = df.copy()
    if "decision_time" not in output.columns:
        output["decision_time"] = output["timestamp"]
    else:
        output["decision_time"] = pd.to_datetime(output["decision_time"], errors="coerce")
        output["decision_time"] = output["decision_time"].fillna(output["timestamp"])
    effective_horizons = compute_effective_horizons(output, cfg)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else max(1, cfg.adaptive_horizon.base_horizon)
    if max_horizon > 0:
        if len(output) <= max_horizon:
            empty_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
            return output.iloc[0:0][empty_columns].copy()
        output = output.iloc[:-max_horizon].copy()

    output_columns = BASE_OUTPUT_COLUMNS + feature_columns + BARRIER_OUTPUT_COLUMNS + ["Target"]
    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    output.dropna(inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output
