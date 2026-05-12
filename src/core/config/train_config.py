from dataclasses import dataclass
from pathlib import Path

from core.config.base import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    dataset_dir: str
    models_dir: str
    model_name: str
    seed: int
    n_splits: int
    split_mode: str
    monthly_train_months: int
    monthly_test_months: int
    monthly_window_mode: str
    purge_gap: int
    use_symbol_feature: bool
    enable_feature_clip: bool
    feature_clip_lower_q: float
    feature_clip_upper_q: float
    sample_weight_half_life_days: float
    sample_weight_min: float
    sample_weight_max: float
    regime_aware_weighting: bool
    regime_recent_days_boost: float
    regime_recent_boost_factor: float
    regime_weight_strength: float
    regime_weight_strength_cap: float
    regime_weight_slope_scale_4h: float
    internal_eval_min_fraction: float
    internal_eval_max_fraction: float
    internal_eval_step_fraction: float
    internal_eval_max_class_rate_diff: float
    early_stopping_rounds: int
    unstable_fold_min_best_iter: int
    unstable_fold_fallback_min_estimators: int
    unstable_fold_fallback_default_estimators: int
    confidence_threshold: float
    train_feature_subset: str  # TRAIN_FEATURE_SUBSET: пусто = все числовые; legacy_mvp_v1 = 21 фичи

    @property
    def models_dir_path(self) -> Path:
        return Path(self.models_dir)

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            symbols=cls.env_list("SYMBOLS", "BTC/USDT"),
            model_path=cls.env_str("MODEL_PATH", "models/latest.pkl"),
            timeframe=cls.env_str("TIMEFRAME", "1h"),
            htf_timeframe=cls.env_str("HTF_TIMEFRAME", "4h"),
            dataset_dir=cls.env_str("DATASET_DIR", "data/labeled/source=bybit"),
            models_dir=cls.env_str("MODELS_DIR", "models"),
            model_name=cls.env_str("TRAIN_MODEL_NAME", "lightgbm_target"),
            seed=int(cls.env_str("TRAIN_SEED", "42")),
            n_splits=int(cls.env_str("TRAIN_N_SPLITS", "5")),
            split_mode=cls.env_str("TRAIN_SPLIT_MODE", "monthly"),
            monthly_train_months=int(cls.env_str("TRAIN_MONTHLY_TRAIN_MONTHS", "6")),
            monthly_test_months=int(cls.env_str("TRAIN_MONTHLY_TEST_MONTHS", "1")),
            monthly_window_mode=cls.env_str("TRAIN_MONTHLY_WINDOW_MODE", "expanding"),
            purge_gap=int(cls.env_str("TRAIN_PURGE_GAP", "24")),
            use_symbol_feature=cls.env_str("USE_SYMBOL_FEATURE", "1") == "1",
            enable_feature_clip=cls.env_str("ENABLE_FEATURE_CLIP", "0") == "1",
            feature_clip_lower_q=float(cls.env_str("FEATURE_CLIP_LOWER_Q", "0.01")),
            feature_clip_upper_q=float(cls.env_str("FEATURE_CLIP_UPPER_Q", "0.99")),
            sample_weight_half_life_days=float(cls.env_str("SAMPLE_WEIGHT_HALF_LIFE_DAYS", "180")),
            sample_weight_min=float(cls.env_str("SAMPLE_WEIGHT_MIN", "0.8")),
            sample_weight_max=float(cls.env_str("SAMPLE_WEIGHT_MAX", "1.35")),
            regime_aware_weighting=cls.env_str("REGIME_AWARE_WEIGHTING", "1") == "1",
            regime_recent_days_boost=float(cls.env_str("REGIME_RECENT_DAYS_BOOST", "60")),
            regime_recent_boost_factor=float(cls.env_str("REGIME_RECENT_BOOST_FACTOR", "1.5")),
            regime_weight_strength=float(cls.env_str("REGIME_WEIGHT_STRENGTH", "0.18")),
            regime_weight_strength_cap=float(cls.env_str("REGIME_WEIGHT_STRENGTH_CAP", "0.25")),
            regime_weight_slope_scale_4h=float(cls.env_str("REGIME_WEIGHT_SLOPE_SCALE_4H", "0.08")),
            internal_eval_min_fraction=float(cls.env_str("INTERNAL_EVAL_MIN_FRACTION", "0.15")),
            internal_eval_max_fraction=float(cls.env_str("INTERNAL_EVAL_MAX_FRACTION", "0.40")),
            internal_eval_step_fraction=float(cls.env_str("INTERNAL_EVAL_STEP_FRACTION", "0.05")),
            internal_eval_max_class_rate_diff=float(cls.env_str("INTERNAL_EVAL_MAX_CLASS_RATE_DIFF", "0.08")),
            early_stopping_rounds=int(cls.env_str("EARLY_STOPPING_ROUNDS", "200")),
            unstable_fold_min_best_iter=int(cls.env_str("UNSTABLE_FOLD_MIN_BEST_ITER", "25")),
            unstable_fold_fallback_min_estimators=int(cls.env_str("UNSTABLE_FOLD_FALLBACK_MIN_ESTIMATORS", "150")),
            unstable_fold_fallback_default_estimators=int(
                cls.env_str("UNSTABLE_FOLD_FALLBACK_DEFAULT_ESTIMATORS", "250")
            ),
            confidence_threshold=float(cls.env_str("CONFIDENCE_THRESHOLD", "0.55")),
            train_feature_subset=cls.env_str("TRAIN_FEATURE_SUBSET", "").strip(),
        )
