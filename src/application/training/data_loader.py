from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_training_frame(dataset_dir: str, symbols: list[str], timeframe: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    root = Path(dataset_dir)
    for symbol in symbols:
        symbol_key = symbol.replace("/", "")
        path = root / f"symbol={symbol_key}" / f"timeframe={timeframe}" / "dataset.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "decision_time" not in out.columns:
        out["decision_time"] = out["timestamp"]
    out["decision_time"] = pd.to_datetime(out["decision_time"], errors="coerce")
    out["decision_time"] = out["decision_time"].fillna(out["timestamp"])
    out = out.dropna(subset=["timestamp", "Target"]).sort_values("timestamp").reset_index(drop=True)
    out.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    return out
