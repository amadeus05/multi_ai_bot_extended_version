from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sample_weights(
    frame: pd.DataFrame,
    half_life_days: float,
    min_weight: float,
    max_weight: float,
    regime_aware: bool,
    recent_days: float,
    recent_boost_factor: float,
    regime_weight_strength: float,
    regime_weight_strength_cap: float,
    regime_weight_slope_scale_4h: float,
) -> np.ndarray:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days_ago = (ts.max() - ts).dt.total_seconds() / 86400.0
    decay = np.log(2) / max(half_life_days, 1.0)
    weights = np.exp(-decay * days_ago.fillna(0).to_numpy())
    if regime_aware:
        recent_mask = days_ago <= recent_days
        weights = np.where(recent_mask, weights * recent_boost_factor, weights)
        regime_score = np.zeros(len(frame), dtype=float)
        regime_terms = 0
        if "market_breadth_ema_fast_slow_1h" in frame.columns:
            breadth_1h = pd.to_numeric(frame["market_breadth_ema_fast_slow_1h"], errors="coerce").fillna(0.5)
            regime_score += np.abs((breadth_1h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
            regime_terms += 1
        if "market_breadth_pos_return_4h_3" in frame.columns:
            breadth_4h = pd.to_numeric(frame["market_breadth_pos_return_4h_3"], errors="coerce").fillna(0.5)
            regime_score += np.abs((breadth_4h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
            regime_terms += 1
        if "ema_slope_4h" in frame.columns:
            ema_slope_4h = pd.to_numeric(frame["ema_slope_4h"], errors="coerce").fillna(0.0)
            if regime_weight_slope_scale_4h > 0:
                regime_score += np.tanh(np.abs(ema_slope_4h.to_numpy()) / regime_weight_slope_scale_4h)
                regime_terms += 1
        if regime_terms > 0:
            regime_score /= float(regime_terms)
            regime_strength = np.clip(regime_score * regime_weight_strength, 0.0, regime_weight_strength_cap)
            weights = weights * (1.0 + regime_strength)
    return np.clip(weights, min_weight, max_weight)
