from __future__ import annotations

from domain.ml.features import config as cfg
import pandas as pd

from domain.ml.features.builders.base_cross_sectional_feature_builder import BaseCrossSectionalFeatureBuilder
from domain.ml.features.builders.base_market_context_feature_builder import BaseMarketContextFeatureBuilder
from domain.ml.features.builders.btc_relative_feature_builder import BtcRelativeFeatureBuilder
from domain.ml.features.builders.funding_feature_builder import FundingFeatureBuilder
from domain.ml.features.builders.htf_cross_sectional_feature_builder import HtfCrossSectionalFeatureBuilder
from domain.ml.features.builders.htf_feature_builder import HtfFeatureBuilder
from domain.ml.features.builders.htf_market_context_feature_builder import HtfMarketContextFeatureBuilder
from domain.ml.features.builders.interaction_feature_builder import InteractionFeatureBuilder
from domain.ml.features.builders.momentum_feature_builder import MomentumFeatureBuilder
from domain.ml.features.builders.open_interest_feature_builder import OpenInterestFeatureBuilder
from domain.ml.features.builders.premium_feature_builder import PremiumFeatureBuilder
from domain.ml.features.builders.regime_feature_builder import RegimeFeatureBuilder
from domain.ml.features.builders.structure_feature_builder import StructureFeatureBuilder
from domain.ml.features.builders.time_context_feature_builder import TimeContextFeatureBuilder
from domain.ml.features.builders.base_feature_builder import FeatureBuilder
from domain.ml.features.models.feature_context import FeatureContext
from domain.ml.features.models.feature_pipeline_result import FeaturePipelineResult
from domain.ml.features.models.feature_request import ResolvedFeatureRequest, resolve_feature_request
from domain.ml.features.models.indicator_cache import IndicatorCache
from domain.ml.features.models.feature_spec import FeatureSpec


class MasterFeatureBuilder:
    def __init__(self) -> None:
        self.main_builders: list[FeatureBuilder] = [
            MomentumFeatureBuilder(),
            RegimeFeatureBuilder(),
            StructureFeatureBuilder(),
            TimeContextFeatureBuilder(),
            FundingFeatureBuilder(),
            PremiumFeatureBuilder(),
            OpenInterestFeatureBuilder(),
        ]
        self.htf_builders: list[FeatureBuilder] = [
            HtfFeatureBuilder(),
        ]
        self.base_enrichment_builders: list[FeatureBuilder] = [
            BtcRelativeFeatureBuilder(),
            BaseCrossSectionalFeatureBuilder(),
            BaseMarketContextFeatureBuilder(),
        ]
        self.htf_enrichment_builders: list[FeatureBuilder] = [
            HtfCrossSectionalFeatureBuilder(),
            HtfMarketContextFeatureBuilder(),
        ]
        self.post_merge_builders: list[FeatureBuilder] = [
            InteractionFeatureBuilder(),
        ]

    def build(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame],
    ) -> FeaturePipelineResult:
        request = self._resolve_request()
        requested_features = set(request.active_features)
        requested_feature_specs = self.collect_feature_specs(requested_features)

        base_feature_map = self._build_symbol_map(
            candle_map=base_candle_map,
            builders=self.main_builders,
            requested_features=requested_features,
        )
        htf_feature_map = self._build_symbol_map(
            candle_map=htf_candle_map,
            builders=self.htf_builders,
            requested_features=requested_features,
        )

        self._enrich_symbol_map(
            target_map=base_feature_map,
            builders=self.base_enrichment_builders,
            requested_features=requested_features,
            base_feature_map=base_feature_map,
            htf_feature_map=htf_feature_map,
        )
        self._enrich_symbol_map(
            target_map=htf_feature_map,
            builders=self.htf_enrichment_builders,
            requested_features=requested_features,
            base_feature_map=base_feature_map,
            htf_feature_map=htf_feature_map,
        )

        merged_feature_map = self._merge_main_and_htf(
            base_feature_map=base_feature_map,
            htf_feature_map=htf_feature_map,
            requested_features=requested_features,
        )
        final_feature_map = self._apply_post_merge_builders(
            feature_map=merged_feature_map,
            requested_features=requested_features,
        )

        return FeaturePipelineResult(
            feature_map=final_feature_map,
            feature_columns=tuple(sorted(requested_features)),
            active_blocks=request.active_blocks,
            profile_name=request.profile,
            feature_specs=requested_feature_specs,
        )

    def _resolve_request(self) -> ResolvedFeatureRequest:
        block_features = self._collect_block_features()
        raw_request = getattr(cfg, "FEATURE_BUILD_REQUEST", {})
        profile_map = getattr(cfg, "FEATURE_PROFILES", {})
        return resolve_feature_request(raw_request, profile_map, block_features)

    def _collect_block_features(self) -> dict[str, set[str]]:
        block_features: dict[str, set[str]] = {}
        for builder in self._all_builders():
            block_features.setdefault(builder.block_name, set()).update(builder.provides())
        return block_features

    def collect_feature_specs(self, requested_features: set[str] | None = None) -> dict[str, FeatureSpec]:
        feature_specs: dict[str, FeatureSpec] = {}
        for builder in self._all_builders():
            builder_specs = builder.describe_features(requested_features)
            duplicate_features = sorted(set(feature_specs).intersection(builder_specs))
            if duplicate_features:
                raise ValueError(
                    "Duplicate feature specs detected: " + ", ".join(duplicate_features)
                )
            feature_specs.update(builder_specs)
        return feature_specs

    def _all_builders(self) -> list[FeatureBuilder]:
        return [
            *self.main_builders,
            *self.htf_builders,
            *self.base_enrichment_builders,
            *self.htf_enrichment_builders,
            *self.post_merge_builders,
        ]

    def _build_symbol_map(
        self,
        candle_map: dict[str, pd.DataFrame],
        builders: list[FeatureBuilder],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        built_map: dict[str, pd.DataFrame] = {}
        for symbol, source_df in candle_map.items():
            frame = source_df.copy().sort_values("timestamp").reset_index(drop=True)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").astype("datetime64[ns]")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            output = frame.copy()
            context = FeatureContext(
                frame=frame,
                symbol=symbol,
                indicator_cache=IndicatorCache(),
            )
            for builder in builders:
                block_request = builder.provides().intersection(requested_features)
                if not block_request:
                    continue
                feature_block = builder.build(context, block_request)
                output = self._merge_feature_block(output, feature_block)
            built_map[symbol] = output
        return built_map

    def _enrich_symbol_map(
        self,
        target_map: dict[str, pd.DataFrame],
        builders: list[FeatureBuilder],
        requested_features: set[str],
        base_feature_map: dict[str, pd.DataFrame],
        htf_feature_map: dict[str, pd.DataFrame],
    ) -> None:
        shared_cache: dict[str, object] = {}
        for symbol, source_df in list(target_map.items()):
            output = source_df.copy()
            context = FeatureContext(
                frame=source_df,
                symbol=symbol,
                base_feature_map=base_feature_map,
                htf_feature_map=htf_feature_map,
                shared_cache=shared_cache,
            )
            for builder in builders:
                block_request = builder.provides().intersection(requested_features)
                if not block_request:
                    continue
                feature_block = builder.build(context, block_request)
                output = self._merge_feature_block(output, feature_block)
            target_map[symbol] = output

    def _merge_main_and_htf(
        self,
        base_feature_map: dict[str, pd.DataFrame],
        htf_feature_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        merged_map: dict[str, pd.DataFrame] = {}
        htf_features = set().union(*(builder.provides() for builder in self.htf_builders + self.htf_enrichment_builders))
        htf_requested_columns = [column for column in sorted(htf_features) if column in requested_features]

        for symbol, base_df in base_feature_map.items():
            output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
            output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce").astype("datetime64[ns]")
            output = output.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            htf_df = htf_feature_map.get(symbol)
            if htf_df is None or htf_df.empty:
                for column in htf_requested_columns:
                    output[column] = pd.NA
                merged_map[symbol] = output
                continue

            merge_columns = ["timestamp"] + [column for column in htf_requested_columns if column in htf_df.columns]
            htf_merge_frame = htf_df[merge_columns].copy()
            htf_merge_frame["timestamp"] = pd.to_datetime(htf_merge_frame["timestamp"], errors="coerce").astype("datetime64[ns]")
            htf_merge_frame = htf_merge_frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            merged = pd.merge_asof(
                output,
                htf_merge_frame,
                on="timestamp",
                direction="backward",
            )
            for column in htf_requested_columns:
                if column not in merged.columns:
                    merged[column] = pd.NA
            merged_map[symbol] = merged
        return merged_map

    def _apply_post_merge_builders(
        self,
        feature_map: dict[str, pd.DataFrame],
        requested_features: set[str],
    ) -> dict[str, pd.DataFrame]:
        shared_cache: dict[str, object] = {}
        output_map: dict[str, pd.DataFrame] = {}
        for symbol, source_df in feature_map.items():
            output = source_df.copy()
            context = FeatureContext(
                frame=output,
                symbol=symbol,
                base_feature_map=feature_map,
                shared_cache=shared_cache,
            )
            for builder in self.post_merge_builders:
                block_request = builder.provides().intersection(requested_features)
                if not block_request:
                    continue
                feature_block = builder.build(context, block_request)
                output = self._merge_feature_block(output, feature_block)
            output_map[symbol] = output
        return output_map

    @staticmethod
    def _merge_feature_block(base_df: pd.DataFrame, feature_block: pd.DataFrame) -> pd.DataFrame:
        if feature_block.empty or list(feature_block.columns) == ["timestamp"]:
            return base_df
        extra_columns = [column for column in feature_block.columns if column != "timestamp"]
        deduped_block = feature_block.loc[:, ["timestamp"] + extra_columns]
        return base_df.merge(deduped_block, on="timestamp", how="left")
