from __future__ import annotations

import pandas as pd


def build_feature_clip_bounds(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    enabled: bool,
    lower_q: float,
    upper_q: float,
) -> dict[str, dict[str, float]]:
    if not enabled:
        return {}
    bounds: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        if column == "symbol":
            continue
        series = train_df[column].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if series.empty:
            continue
        lo = series.quantile(lower_q)
        hi = series.quantile(upper_q)
        if pd.isna(lo) or pd.isna(hi):
            continue
        bounds[column] = {"lower": float(lo), "upper": float(hi)}
    return bounds


def apply_feature_clip_bounds(frame: pd.DataFrame, bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not bounds:
        return frame
    out = frame.copy()
    for column, clip in bounds.items():
        if column in out.columns:
            out[column] = out[column].clip(lower=clip["lower"], upper=clip["upper"])
    return out
