from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config.train_config import TrainConfig


def build_model(seed: int, n_estimators: int = 800) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.005,
        num_leaves=15,
        min_child_samples=150,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_split_gain=0.01,
        subsample_freq=1,
    )


def fit_model_with_internal_eval(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    feature_columns: list[str],
    seed: int,
    cfg: TrainConfig,
    best_iterations_so_far: list[int] | None = None,
) -> tuple[lgb.LGBMClassifier, int, dict]:
    model = build_model(seed)
    if len(set(y_train.unique().tolist())) < 2:
        raise RuntimeError("Training fold contains only one class.")
    eval_size = _resolve_internal_eval_size(y_train, cfg)
    eval_size = min(eval_size, max(1, len(y_train) - 1))
    x_fit = x_train.iloc[:-eval_size]
    y_fit = y_train.iloc[:-eval_size]
    w_fit = w_train[:-eval_size]
    x_eval = x_train.iloc[-eval_size:]
    y_eval = y_train.iloc[-eval_size:]
    if len(set(y_fit.unique().tolist())) < 2 or len(set(y_eval.unique().tolist())) < 2:
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            categorical_feature=["symbol"] if "symbol" in feature_columns else "auto",
        )
    else:
        model.fit(
            x_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(x_eval, y_eval)],
            eval_metric="binary_logloss",
            categorical_feature=["symbol"] if "symbol" in feature_columns else "auto",
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    best_iter = int(model.best_iteration_ or model.n_estimators_)
    fallback_used = False
    fallback_n = None
    if best_iter < cfg.unstable_fold_min_best_iter:
        prior_median = int(np.median(best_iterations_so_far)) if best_iterations_so_far else cfg.unstable_fold_fallback_default_estimators
        fallback_n = max(cfg.unstable_fold_fallback_min_estimators, prior_median)
        model = build_model(seed=seed, n_estimators=fallback_n)
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            categorical_feature=["symbol"] if "symbol" in feature_columns else "auto",
        )
        best_iter = int(model.n_estimators_)
        fallback_used = True
    fit_meta = {
        "internal_eval_size": int(eval_size),
        "eval_rows": int(eval_size),
        "fit_pos_rate": float(y_fit.mean()) if len(y_fit) > 0 else 0.0,
        "eval_pos_rate": float(y_eval.mean()) if len(y_eval) > 0 else 0.0,
        "fallback_used": fallback_used,
        "fallback_n_estimators": int(fallback_n) if fallback_n is not None else None,
    }
    return model, best_iter, fit_meta


def _resolve_internal_eval_size(y_train: pd.Series, cfg: TrainConfig) -> int:
    n_rows = int(len(y_train))
    if n_rows <= 1:
        return 1
    min_fraction = min(max(cfg.internal_eval_min_fraction, 0.05), 0.45)
    max_fraction = min(max(cfg.internal_eval_max_fraction, min_fraction), 0.50)
    step_fraction = min(max(cfg.internal_eval_step_fraction, 0.01), 0.10)
    candidate_fractions = []
    frac = min_fraction
    while frac <= max_fraction + 1e-9:
        candidate_fractions.append(round(frac, 4))
        frac += step_fraction
    best_eval_size = max(1, int(n_rows * min_fraction))
    best_diff = None
    for fraction in candidate_fractions:
        eval_size = max(1, int(n_rows * fraction))
        if eval_size >= n_rows:
            eval_size = n_rows - 1
        if eval_size <= 0:
            continue
        y_fit = y_train.iloc[:-eval_size]
        y_eval = y_train.iloc[-eval_size:]
        if y_fit.empty:
            continue
        if len(set(y_fit.unique().tolist())) < 2 or len(set(y_eval.unique().tolist())) < 2:
            continue
        diff = abs(float(y_eval.mean()) - float(y_fit.mean()))
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_eval_size = eval_size
        if diff <= cfg.internal_eval_max_class_rate_diff:
            return eval_size
    return best_eval_size
