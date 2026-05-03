from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketContextSpec:
    name: str
    column: str


class MarketContextAssembler:
    """Shared timestamp normalization and as-of context attachment for train/live frames."""

    OPEN_INTEREST = MarketContextSpec("open interest", "open_interest")
    FUNDING = MarketContextSpec("funding", "funding_rate")
    PREMIUM_INDEX = MarketContextSpec("premium index", "premium_index_close")

    @staticmethod
    def normalize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        out = frame.copy()
        if "timestamp" not in out.columns:
            return out
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    @classmethod
    def dedupe_rows(cls, frame: pd.DataFrame, *, subset: tuple[str, ...] = ("timestamp",)) -> pd.DataFrame:
        out = cls.normalize_timestamps(frame)
        if out.empty:
            return out
        return out.drop_duplicates(subset=list(subset), keep="last").reset_index(drop=True)

    @classmethod
    def merge_asof_context(
        cls,
        base: pd.DataFrame,
        context: pd.DataFrame,
        value_col: str,
        *,
        add_missing_column: bool,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        out = cls.normalize_timestamps(base)
        if out.empty:
            return out
        ctx = cls.normalize_timestamps(context)
        if ctx.empty or value_col not in ctx.columns:
            if add_missing_column:
                out[value_col] = fill_value
            return out
        merged = pd.merge_asof(
            out,
            ctx[["timestamp", value_col]],
            on="timestamp",
            direction="backward",
        )
        merged[value_col] = merged[value_col].ffill().fillna(fill_value)
        return merged

    @classmethod
    def attach_contexts(
        cls,
        base: pd.DataFrame,
        contexts: dict[str, pd.DataFrame],
        *,
        add_missing_columns: bool,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        out = cls.normalize_timestamps(base)
        for value_col, context in contexts.items():
            out = cls.merge_asof_context(
                out,
                context,
                value_col,
                add_missing_column=add_missing_columns,
                fill_value=fill_value,
            )
        return out
