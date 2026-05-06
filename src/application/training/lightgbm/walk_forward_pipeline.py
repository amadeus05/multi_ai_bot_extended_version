from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from application.training.common.feature_selection import (
    filter_feature_columns_to_allowlist,
    resolve_train_feature_allowlist,
    select_feature_columns,
)
from application.training.common.splits import build_timestamp_splits
from application.training.common.weights import compute_sample_weights
from application.training.lightgbm.feature_clipping import apply_feature_clip_bounds, build_feature_clip_bounds
from application.training.lightgbm.trainer import fit_model_with_internal_eval
from core.config.train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    fold_details: list[dict]
    feature_columns: list[str]


class WalkForwardPipeline:
    """Walk-forward OOS: на каждом фолде обучение на train, predict_proba на test, склейка предиктов."""

    def __init__(self, *, train_cfg: TrainConfig) -> None:
        self.train_cfg = train_cfg

    def run(self, dataset: pd.DataFrame) -> WalkForwardResult:
        if dataset.empty:
            raise RuntimeError("Walk-forward: пустой датасет.")

        raw_labels = dataset["Target"].astype(int)
        dataset = dataset.loc[raw_labels != 0].copy()
        dataset["Target"] = dataset["Target"].astype(int).map({-1: 0, 1: 1})
        dataset["symbol"] = dataset["symbol"].astype("category")

        feature_columns = select_feature_columns(dataset, use_symbol_feature=self.train_cfg.use_symbol_feature)
        allowlist = resolve_train_feature_allowlist(self.train_cfg.train_feature_subset)
        if allowlist is not None:
            feature_columns = filter_feature_columns_to_allowlist(dataset, allowlist)

        unique_ts = np.sort(dataset["timestamp"].unique())
        splits = build_timestamp_splits(
            unique_ts=unique_ts,
            n_splits=self.train_cfg.n_splits,
            split_mode=self.train_cfg.split_mode,
            monthly_train_months=self.train_cfg.monthly_train_months,
            monthly_test_months=self.train_cfg.monthly_test_months,
            monthly_window_mode=self.train_cfg.monthly_window_mode,
        )
        if not splits:
            raise RuntimeError("Walk-forward: нет валидных сплитов.")

        prediction_frames: list[pd.DataFrame] = []
        fold_details: list[dict] = []
        best_iterations: list[int] = []

        for fold_idx, original_train_ts, test_ts in splits:
            train_ts = original_train_ts
            if self.train_cfg.purge_gap > 0 and len(train_ts) > self.train_cfg.purge_gap:
                train_ts = train_ts[: -self.train_cfg.purge_gap]
            train_df = dataset.loc[dataset["timestamp"].isin(set(train_ts))].copy()
            test_df = dataset.loc[dataset["timestamp"].isin(set(test_ts))].copy()
            if train_df.empty or test_df.empty:
                continue
            if len(sorted(train_df["Target"].unique().tolist())) < 2:
                continue

            clip_bounds = build_feature_clip_bounds(
                train_df,
                feature_columns,
                enabled=self.train_cfg.enable_feature_clip,
                lower_q=self.train_cfg.feature_clip_lower_q,
                upper_q=self.train_cfg.feature_clip_upper_q,
            )
            train_df = apply_feature_clip_bounds(train_df, clip_bounds)
            test_df = apply_feature_clip_bounds(test_df, clip_bounds)

            x_train = train_df[feature_columns]
            y_train = train_df["Target"]
            x_test = test_df[feature_columns]
            y_test = test_df["Target"]
            w_train = compute_sample_weights(
                frame=train_df,
                half_life_days=self.train_cfg.sample_weight_half_life_days,
                min_weight=self.train_cfg.sample_weight_min,
                max_weight=self.train_cfg.sample_weight_max,
                regime_aware=self.train_cfg.regime_aware_weighting,
                recent_days=self.train_cfg.regime_recent_days_boost,
                recent_boost_factor=self.train_cfg.regime_recent_boost_factor,
                regime_weight_strength=self.train_cfg.regime_weight_strength,
                regime_weight_strength_cap=self.train_cfg.regime_weight_strength_cap,
                regime_weight_slope_scale_4h=self.train_cfg.regime_weight_slope_scale_4h,
            )
            model, best_iter, _fit_meta = fit_model_with_internal_eval(
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                feature_columns=feature_columns,
                seed=self.train_cfg.seed,
                cfg=self.train_cfg,
                best_iterations_so_far=best_iterations,
            )
            best_iterations.append(best_iter)

            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            fold_acc = float((y_pred == y_test.to_numpy()).mean())
            fold_auc = float(roc_auc_score(y_test, y_proba[:, 1]))

            pred = test_df[["timestamp", "symbol"]].copy()
            pred["symbol"] = pred["symbol"].astype(str)
            pred["p_short"] = y_proba[:, 0].astype(float)
            pred["p_long"] = y_proba[:, 1].astype(float)
            pred["fold"] = int(fold_idx)
            prediction_frames.append(pred)

            fold_details.append(
                {
                    "fold": int(fold_idx),
                    "split_mode": self.train_cfg.split_mode,
                    "train_rows": int(len(train_df)),
                    "prediction_rows": int(len(test_df)),
                    "test_rows": int(len(test_df)),
                    "purged_timestamps": int(self.train_cfg.purge_gap),
                    "train_start": str(train_df["timestamp"].iloc[0]),
                    "train_end": str(train_df["timestamp"].iloc[-1]),
                    "test_start": str(test_df["timestamp"].iloc[0]),
                    "test_end": str(test_df["timestamp"].iloc[-1]),
                    "best_iteration": int(best_iter),
                    "accuracy": fold_acc,
                    "roc_auc": fold_auc,
                }
            )
            logger.info(
                "Walk-forward fold %s | train=%s test=%s | best_iter=%s acc=%.4f auc=%.4f",
                fold_idx,
                len(train_df),
                len(test_df),
                best_iter,
                fold_acc,
                fold_auc,
            )

        if not prediction_frames:
            raise RuntimeError("Walk-forward: все фолды пропущены, предиктов нет.")

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
        predictions = predictions.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        return WalkForwardResult(
            predictions=predictions,
            fold_details=fold_details,
            feature_columns=feature_columns,
        )

    @staticmethod
    def save_artifacts(predictions: pd.DataFrame, fold_details: list[dict], train_cfg: TrainConfig) -> Path:
        out_dir = train_cfg.models_dir_path
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "walk_forward_oos_predictions.csv"
        predictions.to_csv(csv_path, index=False)

        ts_min = predictions["timestamp"].min()
        ts_max = predictions["timestamp"].max()
        summary = {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbols": list(train_cfg.symbols),
            "prediction_rows": int(len(predictions)),
            "prediction_period": {
                "start": str(ts_min),
                "end": str(ts_max),
            },
            "fold_details": fold_details,
        }
        json_path = out_dir / "walk_forward_oos_predictions_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Walk-forward артефакты: %s, %s", csv_path, json_path)
        return csv_path
