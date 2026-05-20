from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from application.modeling.contracts import FoldData, FoldPrediction, ModelingDataset
from application.training.feature_clipping import apply_feature_clip_bounds, build_feature_clip_bounds
from application.training.feature_selection import (
    filter_feature_columns_to_allowlist,
    resolve_train_feature_allowlist,
    select_feature_columns,
)
from application.training.lgbm_trainer import fit_model_with_internal_eval
from application.training.weights import compute_sample_weights
from core.config.train_config import TrainConfig

logger = logging.getLogger(__name__)


class LightGbmWalkForwardAdapter:
    model_id = "lightgbm"

    def __init__(self, *, train_cfg: TrainConfig) -> None:
        self._cfg = train_cfg
        self._best_iterations: list[int] = []

    @property
    def best_iterations(self) -> list[int]:
        return list(self._best_iterations)

    def prepare_dataset(self, dataset: pd.DataFrame) -> ModelingDataset:
        if dataset.empty:
            raise RuntimeError("Walk-forward: empty dataset.")

        self._best_iterations = []
        full_prediction_frame = dataset.copy()
        full_prediction_frame["Target"] = full_prediction_frame["Target"].astype(int)
        full_prediction_frame["symbol"] = full_prediction_frame["symbol"].astype("category")

        raw_labels = dataset["Target"].astype(int)
        prepared = dataset.loc[raw_labels != 0].copy()
        prepared["Target"] = prepared["Target"].astype(int).map({-1: 0, 1: 1})
        prepared["symbol"] = prepared["symbol"].astype("category")

        feature_columns = select_feature_columns(prepared, use_symbol_feature=self._cfg.use_symbol_feature)
        allowlist = resolve_train_feature_allowlist(self._cfg.train_feature_subset)
        if allowlist is not None:
            feature_columns = filter_feature_columns_to_allowlist(prepared, allowlist)

        return ModelingDataset(
            frame=prepared,
            feature_columns=list(feature_columns),
            unique_timestamps=np.sort(full_prediction_frame["timestamp"].dropna().unique()),
            prediction_frame=full_prediction_frame,
        )

    def should_skip_fold(self, fold: FoldData) -> bool:
        if fold.train_frame.empty or fold.test_frame.empty:
            return True
        return len(sorted(fold.train_frame["Target"].unique().tolist())) < 2

    def fit_predict_fold(self, fold: FoldData) -> FoldPrediction:
        train_df = fold.train_frame
        test_df = fold.test_frame
        prediction_test_df = fold.prediction_test_frame if fold.prediction_test_frame is not None else test_df
        feature_columns = fold.feature_columns

        clip_bounds = build_feature_clip_bounds(
            train_df,
            feature_columns,
            enabled=self._cfg.enable_feature_clip,
            lower_q=self._cfg.feature_clip_lower_q,
            upper_q=self._cfg.feature_clip_upper_q,
        )
        train_df = apply_feature_clip_bounds(train_df, clip_bounds)
        test_df = apply_feature_clip_bounds(test_df, clip_bounds)
        prediction_test_df = apply_feature_clip_bounds(prediction_test_df, clip_bounds)

        x_train = train_df[feature_columns]
        y_train = train_df["Target"]
        x_test = test_df[feature_columns]
        y_test = test_df["Target"]
        x_prediction = prediction_test_df[feature_columns]
        w_train = compute_sample_weights(
            frame=train_df,
            half_life_days=self._cfg.sample_weight_half_life_days,
            min_weight=self._cfg.sample_weight_min,
            max_weight=self._cfg.sample_weight_max,
            regime_aware=self._cfg.regime_aware_weighting,
            recent_days=self._cfg.regime_recent_days_boost,
            recent_boost_factor=self._cfg.regime_recent_boost_factor,
            regime_weight_strength=self._cfg.regime_weight_strength,
            regime_weight_strength_cap=self._cfg.regime_weight_strength_cap,
            regime_weight_slope_scale_4h=self._cfg.regime_weight_slope_scale_4h,
        )
        model, best_iter, _fit_meta = fit_model_with_internal_eval(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            feature_columns=feature_columns,
            seed=self._cfg.seed,
            cfg=self._cfg,
            best_iterations_so_far=self._best_iterations,
        )
        self._best_iterations.append(best_iter)

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        prediction_proba = model.predict_proba(x_prediction)
        fold_acc = float((y_pred == y_test.to_numpy()).mean())
        fold_auc = float(roc_auc_score(y_test, y_proba[:, 1]))

        prediction_columns = ["timestamp", "symbol"]
        if "decision_time" in prediction_test_df.columns:
            prediction_columns.append("decision_time")
        predictions = prediction_test_df[prediction_columns].copy()
        predictions["symbol"] = predictions["symbol"].astype(str)
        predictions["p_short"] = prediction_proba[:, 0].astype(float)
        predictions["p_long"] = prediction_proba[:, 1].astype(float)
        predictions["fold"] = int(fold.fold_idx)

        details = {
            "fold": int(fold.fold_idx),
            "split_mode": self._cfg.split_mode,
            "train_rows": int(len(train_df)),
            "prediction_rows": int(len(prediction_test_df)),
            "test_rows": int(len(test_df)),
            "all_test_rows": int(len(prediction_test_df)),
            "purged_timestamps": int(self._cfg.purge_gap),
            "train_start": str(train_df["timestamp"].iloc[0]),
            "train_end": str(train_df["timestamp"].iloc[-1]),
            "test_start": str(test_df["timestamp"].iloc[0]),
            "test_end": str(test_df["timestamp"].iloc[-1]),
            "best_iteration": int(best_iter),
            "accuracy": fold_acc,
            "roc_auc": fold_auc,
        }
        logger.info(
            "Walk-forward fold %s | model=%s | train=%s test=%s | best_iter=%s acc=%.4f auc=%.4f",
            fold.fold_idx,
            self.model_id,
            len(train_df),
            len(test_df),
            best_iter,
            fold_acc,
            fold_auc,
        )
        return FoldPrediction(
            predictions=predictions,
            fold_details=details,
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
        )
