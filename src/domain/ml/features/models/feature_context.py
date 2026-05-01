from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from domain.ml.features.models.indicator_cache import IndicatorCache


@dataclass(slots=True)
class FeatureContext:
    frame: pd.DataFrame
    symbol: str
    base_feature_map: dict[str, pd.DataFrame] | None = None
    htf_feature_map: dict[str, pd.DataFrame] | None = None
    indicator_cache: IndicatorCache = field(default_factory=IndicatorCache)
    shared_cache: dict[str, Any] = field(default_factory=dict)
