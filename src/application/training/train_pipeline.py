from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from application.training.artifacts import save_artifacts
from application.training.data_loader import load_training_frame
from application.training.feature_clipping import apply_feature_clip_bounds, build_feature_clip_bounds
from application.training.feature_selection import (
    filter_feature_columns_to_allowlist,
    resolve_train_feature_allowlist,
    select_feature_columns,
)
from application.training.lgbm_trainer import build_model, fit_model_with_internal_eval
from application.training.metrics import evaluate_model
from application.training.splits import build_timestamp_splits
from application.training.weights import compute_sample_weights
from core.config.train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    oos_metrics: dict
    fold_details: list[dict]
    median_best_iteration: int
    feature_columns: list[str]
    artifact_paths: dict[str, str]


class TrainPipeline:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def run(self) -> TrainingResult:
        dataset = load_training_frame(self.cfg.dataset_dir, self.cfg.symbols, self.cfg.timeframe)
        if dataset.empty:
            raise RuntimeError("Training dataset is empty. Run dataset pipeline first.")

        raw_labels = dataset["Target"].astype(int)
        all_timestamps = np.sort(dataset["timestamp"].dropna().unique())
        dataset = dataset.loc[raw_labels != 0].copy()
        dataset["Target"] = dataset["Target"].astype(int).map({-1: 0, 1: 1})
        dataset["symbol"] = dataset["symbol"].astype("category")

        feature_columns = select_feature_columns(dataset, use_symbol_feature=self.cfg.use_symbol_feature)
        allowlist = resolve_train_feature_allowlist(self.cfg.train_feature_subset)
        if allowlist is not None:
            feature_columns = filter_feature_columns_to_allowlist(dataset, allowlist)
            logger.info(
                "Feature subset=%s | training_columns=%s",
                self.cfg.train_feature_subset or "(none)",
                len(feature_columns),
            )
        unique_ts = np.sort(dataset["timestamp"].unique())
        splits = build_timestamp_splits(
            unique_ts=unique_ts,
            n_splits=self.cfg.n_splits,
            split_mode=self.cfg.split_mode,
            monthly_train_months=self.cfg.monthly_train_months,
            monthly_test_months=self.cfg.monthly_test_months,
            monthly_window_mode=self.cfg.monthly_window_mode,
        )
        if not splits:
            raise RuntimeError("No valid walk-forward splits.")
        logger.info("=" * 72)
        logger.info(
            "Walk-forward validation | split_mode=%s | folds=%s | purge_gap=%s | monthly=%sm/%sm (%s)",
            self.cfg.split_mode,
            len(splits),
            self.cfg.purge_gap,
            self.cfg.monthly_train_months,
            self.cfg.monthly_test_months,
            self.cfg.monthly_window_mode,
        )
        logger.info("=" * 72)

        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_y_proba: list[np.ndarray] = []
        fold_details: list[dict] = []
        best_iterations: list[int] = []
        fold_importance_frames: list[pd.DataFrame] = []

        for fold_idx, original_train_ts, test_ts in splits:
            train_ts = original_train_ts
            if self.cfg.purge_gap > 0 and len(train_ts) > self.cfg.purge_gap:
                train_ts = train_ts[:-self.cfg.purge_gap]
            train_df = dataset.loc[dataset["timestamp"].isin(set(train_ts))].copy()
            test_df = dataset.loc[dataset["timestamp"].isin(set(test_ts))].copy()
            if train_df.empty or test_df.empty:
                continue
            if len(sorted(train_df["Target"].unique().tolist())) < 2:
                continue

            clip_bounds = build_feature_clip_bounds(
                train_df,
                feature_columns,
                enabled=self.cfg.enable_feature_clip,
                lower_q=self.cfg.feature_clip_lower_q,
                upper_q=self.cfg.feature_clip_upper_q,
            )
            train_df = apply_feature_clip_bounds(train_df, clip_bounds)
            test_df = apply_feature_clip_bounds(test_df, clip_bounds)

            x_train = train_df[feature_columns]
            y_train = train_df["Target"]
            x_test = test_df[feature_columns]
            y_test = test_df["Target"]
            w_train = compute_sample_weights(
                frame=train_df,
                half_life_days=self.cfg.sample_weight_half_life_days,
                min_weight=self.cfg.sample_weight_min,
                max_weight=self.cfg.sample_weight_max,
                regime_aware=self.cfg.regime_aware_weighting,
                recent_days=self.cfg.regime_recent_days_boost,
                recent_boost_factor=self.cfg.regime_recent_boost_factor,
                regime_weight_strength=self.cfg.regime_weight_strength,
                regime_weight_strength_cap=self.cfg.regime_weight_strength_cap,
                regime_weight_slope_scale_4h=self.cfg.regime_weight_slope_scale_4h,
            )
            model, best_iter, fit_meta = fit_model_with_internal_eval(
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                feature_columns=feature_columns,
                seed=self.cfg.seed,
                cfg=self.cfg,
                best_iterations_so_far=best_iterations,
            )
            best_iterations.append(best_iter)
            fold_importance_frames.append(
                pd.DataFrame(
                    {
                        "feature": feature_columns,
                        f"fold_{fold_idx}_gain": model.booster_.feature_importance(importance_type="gain"),
                        f"fold_{fold_idx}_split": model.booster_.feature_importance(importance_type="split"),
                    }
                )
            )

            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            fold_acc = float((y_pred == y_test.to_numpy()).mean())
            fold_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            all_y_true.append(y_test.to_numpy())
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            fold_details.append(
                {
                    "fold": fold_idx,
                    "split_mode": self.cfg.split_mode,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "best_iteration": int(best_iter),
                    "accuracy": fold_acc,
                    "roc_auc": fold_auc,
                    "purged_timestamps": self.cfg.purge_gap,
                    "timestamp_boundaries": {
                        "train_start": str(train_ts[0]) if len(train_ts) else None,
                        "train_end_before_purge": str(original_train_ts[-1]) if len(original_train_ts) else None,
                        "train_end_after_purge": str(train_ts[-1]) if len(train_ts) else None,
                        "test_start": str(test_ts[0]) if len(test_ts) else None,
                        "test_end": str(test_ts[-1]) if len(test_ts) else None,
                        "train_unique_timestamps_after_purge": int(len(train_ts)),
                        "test_unique_timestamps": int(len(test_ts)),
                    },
                    "train_period": {
                        "start": str(train_df["timestamp"].iloc[0]),
                        "end": str(train_df["timestamp"].iloc[-1]),
                    },
                    "test_period": {
                        "start": str(test_df["timestamp"].iloc[0]),
                        "end": str(test_df["timestamp"].iloc[-1]),
                    },
                    "internal_eval": fit_meta,
                }
            )
            eval_rows = fit_meta.get("eval_rows") if isinstance(fit_meta, dict) else None
            fit_pos_rate = fit_meta.get("fit_pos_rate") if isinstance(fit_meta, dict) else None
            eval_pos_rate = fit_meta.get("eval_pos_rate") if isinstance(fit_meta, dict) else None
            logger.info(
                "Fold %s/%s | train=%s [%s -> %s] | test=%s [%s -> %s] | eval=%s (fit_pos=%.3f eval_pos=%.3f) | best_iter=%s | acc=%.4f | auc=%.4f",
                fold_idx,
                len(splits),
                len(train_df),
                str(train_df["timestamp"].iloc[0]),
                str(train_df["timestamp"].iloc[-1]),
                len(test_df),
                str(test_df["timestamp"].iloc[0]),
                str(test_df["timestamp"].iloc[-1]),
                eval_rows if eval_rows is not None else "-",
                float(fit_pos_rate) if fit_pos_rate is not None else 0.0,
                float(eval_pos_rate) if eval_pos_rate is not None else 0.0,
                best_iter,
                fold_acc,
                fold_auc,
            )

        if not all_y_true:
            raise RuntimeError("All folds were skipped; cannot compute OOS metrics.")

        oos_y_true = np.concatenate(all_y_true)
        oos_y_pred = np.concatenate(all_y_pred)
        oos_y_proba = np.vstack(all_y_proba)
        oos_metrics = evaluate_model(oos_y_true, oos_y_pred, oos_y_proba, confidence_threshold=self.cfg.confidence_threshold)
        median_best_iter = int(np.median(best_iterations))
        logger.info("-" * 72)
        logger.info(
            "OOS aggregate (%s folds, %s rows) | acc=%.4f | bal_acc=%.4f | f1=%.4f | auc=%.4f | pr=%.4f | mcc=%.4f",
            len(fold_details),
            len(oos_y_true),
            oos_metrics["accuracy"],
            oos_metrics["balanced_accuracy"],
            oos_metrics["f1_macro"],
            oos_metrics["roc_auc"],
            oos_metrics["pr_auc"],
            oos_metrics["mcc"],
        )
        logger.info("Median best_iteration across folds: %s", median_best_iter)
        logger.info("=" * 72)

        prod_clip_bounds = build_feature_clip_bounds(
            dataset,
            feature_columns,
            enabled=self.cfg.enable_feature_clip,
            lower_q=self.cfg.feature_clip_lower_q,
            upper_q=self.cfg.feature_clip_upper_q,
        )
        clipped_dataset = apply_feature_clip_bounds(dataset, prod_clip_bounds)
        prod_model = build_model(self.cfg.seed, n_estimators=median_best_iter)
        w_prod = compute_sample_weights(
            frame=clipped_dataset,
            half_life_days=self.cfg.sample_weight_half_life_days,
            min_weight=self.cfg.sample_weight_min,
            max_weight=self.cfg.sample_weight_max,
            regime_aware=self.cfg.regime_aware_weighting,
            recent_days=self.cfg.regime_recent_days_boost,
            recent_boost_factor=self.cfg.regime_recent_boost_factor,
            regime_weight_strength=self.cfg.regime_weight_strength,
            regime_weight_strength_cap=self.cfg.regime_weight_strength_cap,
            regime_weight_slope_scale_4h=self.cfg.regime_weight_slope_scale_4h,
        )
        prod_model.fit(
            clipped_dataset[feature_columns],
            clipped_dataset["Target"],
            sample_weight=w_prod,
            categorical_feature=["symbol"] if "symbol" in feature_columns else "auto",
        )

        fold_importance = pd.DataFrame({"feature": feature_columns})
        for frame in fold_importance_frames:
            fold_importance = fold_importance.merge(frame, on="feature", how="left")
        fold_importance.fillna(0.0, inplace=True)

        metrics_payload = {
            "oos_metrics": oos_metrics,
            "fold_details": fold_details,
            "confidence_threshold": self.cfg.confidence_threshold,
            "fold_stability": {
                "accuracy_std": float(np.std(np.asarray([f["accuracy"] for f in fold_details], dtype=float))) if fold_details else None,
            },
            "median_best_iteration": median_best_iter,
            "total_rows": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "n_splits": self.cfg.n_splits,
            "purge_gap": self.cfg.purge_gap,
            "split_mode": self.cfg.split_mode,
            "monthly_train_months": self.cfg.monthly_train_months,
            "monthly_test_months": self.cfg.monthly_test_months,
            "monthly_window_mode": self.cfg.monthly_window_mode,
            "dataset_diagnostics": {
                "all_timestamps_count": int(len(all_timestamps)),
                "first_all_timestamp": str(all_timestamps[0]) if len(all_timestamps) else None,
                "last_all_timestamp": str(all_timestamps[-1]) if len(all_timestamps) else None,
                "directional_rows_after_filter": int(len(dataset)),
                "fold_boundary_timestamps": [
                    {
                        "fold": int(f["fold"]),
                        "train_rows": int(f["train_rows"]),
                        "test_rows": int(f["test_rows"]),
                        **f.get("timestamp_boundaries", {}),
                    }
                    for f in fold_details
                ],
            },
        }
        artifact_paths = save_artifacts(
            models_dir=self.cfg.models_dir_path,
            model_name=self.cfg.model_name,
            model=prod_model,
            metrics=metrics_payload,
            feature_columns=feature_columns,
            symbols=self.cfg.symbols,
            fold_importance=fold_importance,
            clip_bounds=prod_clip_bounds,
        )
        return TrainingResult(
            oos_metrics=oos_metrics,
            fold_details=fold_details,
            median_best_iteration=median_best_iter,
            feature_columns=feature_columns,
            artifact_paths=artifact_paths,
        )
