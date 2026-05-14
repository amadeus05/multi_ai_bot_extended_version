from __future__ import annotations

import numpy as np
import pandas as pd

from domain.ml.features.builders.base_feature_builder import FeatureBuilder
from domain.ml.features.models.feature_context import FeatureContext
from domain.ml.features.models.feature_spec import feature_spec


class TimeContextFeatureBuilder(FeatureBuilder):
    block_name = "time_context"
    FEATURE_SPECS = {
        "hour_sin_1h": feature_spec(
            "hour_sin_1h",
            block_name,
            "sin(2 * pi * hour(timestamp) / 24)",
            description="Cyclical hour-of-day encoding, sine component.",
            inputs=("timestamp",),
        ),
        "hour_cos_1h": feature_spec(
            "hour_cos_1h",
            block_name,
            "cos(2 * pi * hour(timestamp) / 24)",
            description="Cyclical hour-of-day encoding, cosine component.",
            inputs=("timestamp",),
        ),
        "is_weekend_1h": feature_spec(
            "is_weekend_1h",
            block_name,
            "1{day_of_week(timestamp) >= 5}",
            description="Weekend flag derived from the timestamp.",
            inputs=("timestamp",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.active_features(requested_features)
        frame = context.frame
        output = self.output_frame(context)
        if not active:
            return output

        hour = frame["timestamp"].dt.hour
        if "hour_sin_1h" in active:
            output["hour_sin_1h"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_cos_1h" in active:
            output["hour_cos_1h"] = np.cos(2.0 * np.pi * hour / 24.0)
        if "is_weekend_1h" in active:
            output["is_weekend_1h"] = (frame["timestamp"].dt.dayofweek >= 5).astype(float)
        return output
