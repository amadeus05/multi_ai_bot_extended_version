from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def iter_monthly_timestamp_splits(
    unique_ts: np.ndarray,
    train_months: int,
    test_months: int,
    window_mode: str = "expanding",
):
    timestamps = pd.Series(pd.to_datetime(unique_ts, errors="coerce")).dropna().sort_values()
    if timestamps.empty:
        return
    train_end = timestamps.iloc[0] + pd.DateOffset(months=train_months)
    fold_idx = 1
    while train_end < timestamps.iloc[-1]:
        test_end = train_end + pd.DateOffset(months=test_months)
        if window_mode == "rolling":
            train_start = train_end - pd.DateOffset(months=train_months)
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
        else:
            train_mask = timestamps < train_end
        test_mask = (timestamps >= train_end) & (timestamps < test_end)
        train_ts = timestamps.loc[train_mask].to_numpy()
        test_ts = timestamps.loc[test_mask].to_numpy()
        if len(train_ts) and len(test_ts):
            yield fold_idx, train_ts, test_ts
            fold_idx += 1
        train_end = test_end


def build_timestamp_splits(
    unique_ts: np.ndarray,
    n_splits: int,
    split_mode: str,
    monthly_train_months: int,
    monthly_test_months: int,
    monthly_window_mode: str,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if split_mode == "monthly":
        return list(
            iter_monthly_timestamp_splits(
                unique_ts,
                monthly_train_months,
                monthly_test_months,
                monthly_window_mode,
            )
        )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(idx, unique_ts[tr], unique_ts[te]) for idx, (tr, te) in enumerate(tscv.split(unique_ts), start=1)]
