from __future__ import annotations

import pandas as pd

from core.interfaces.model import Model


def _normalize_timestamp(value) -> pd.Timestamp:
    """Normalize timestamp keys by int64 ns, independent of tz-aware/naive input."""
    ts = pd.Timestamp(value)
    return pd.Timestamp(ts.value)


class OosPredictionReplayModel(Model):
    """Replay precomputed OOS probabilities via the runtime Model interface."""

    def __init__(
        self,
        *,
        lookup: dict[tuple[pd.Timestamp, str], tuple[float, float]],
        feature_columns: list[str],
    ) -> None:
        self._lookup: dict[tuple[pd.Timestamp, str], tuple[float, float]] = {}
        for (ts, sym), probs in lookup.items():
            self._lookup[(_normalize_timestamp(ts), str(sym))] = (float(probs[0]), float(probs[1]))
        self._feature_columns = list(feature_columns)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def required_bars(self) -> int:
        return 1

    def predict(self, features: pd.DataFrame) -> dict:
        if features.empty:
            return {"score": 0.0, "p_long": 0.5, "p_short": 0.5}
        ts = _normalize_timestamp(features["timestamp"].iloc[-1])
        sym = str(features["symbol"].iloc[-1])
        hit = self._lookup.get((ts, sym))
        if hit is None:
            return {"score": 0.0, "p_long": 0.5, "p_short": 0.5}
        p_short, p_long = float(hit[0]), float(hit[1])
        score = (p_long - 0.5) * 2.0
        return {"score": score, "p_long": p_long, "p_short": p_short}


WalkForwardPredictionModel = OosPredictionReplayModel
