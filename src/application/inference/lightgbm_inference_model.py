from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from application.inference.artifacts import (
    load_feature_columns,
    read_json_file,
    resolve_trained_model_paths,
)
from core.interfaces.model import Model

logger = logging.getLogger(__name__)


def load_lightgbm_inference_model(
    *,
    model_path: Path,
    features_path: Path,
    required_bars: int,
) -> Model:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. Обучи через runners/run_train.py или задай MODEL_PATH."
        )
    feature_columns = load_feature_columns(features_path)
    clf = joblib.load(model_path)
    feature_columns = resolve_model_feature_columns(clf, feature_columns, source=str(features_path))
    clip_bounds_path = model_path.parent / f"{model_path.stem}_clip_bounds.json"
    clip_bounds: dict[str, dict[str, float]] | None = None
    if clip_bounds_path.is_file():
        clip_bounds = read_json_file(clip_bounds_path)
    return LightGBMInferenceModel(clf, feature_columns, required_bars=required_bars, clip_bounds=clip_bounds)


def resolve_model_feature_columns(clf, feature_columns: list[str], *, source: str = "features json") -> list[str]:
    model_feature_columns = model_feature_names(clf)
    if not model_feature_columns:
        return list(feature_columns)
    if model_feature_columns != list(feature_columns):
        logger.warning(
            "Feature columns from %s do not match the loaded model; using model feature names "
            "| json_count=%s | model_count=%s | extra_in_json=%s | missing_in_json=%s",
            source,
            len(feature_columns),
            len(model_feature_columns),
            [col for col in feature_columns if col not in model_feature_columns][:10],
            [col for col in model_feature_columns if col not in feature_columns][:10],
        )
        return model_feature_columns
    return list(feature_columns)


def model_feature_names(clf) -> list[str]:
    names = getattr(clf, "feature_name_", None)
    if _is_feature_name_list(names):
        return list(names)
    booster = getattr(clf, "booster_", None) or getattr(clf, "_Booster", None)
    if booster is None or not hasattr(booster, "feature_name"):
        return []
    names = booster.feature_name()
    if _is_feature_name_list(names):
        return list(names)
    return []


def _is_feature_name_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)


def load_default_lightgbm_model(
    *,
    cwd: Path,
    model_path_cfg: str,
    required_bars: int = 250,
    train_model_name: str = "lightgbm_target",
) -> Model:
    """Загружает артефакты из каталога ``models`` (или явный ``MODEL_PATH``), как бэктест/paper/live."""
    models_dir = str((cwd / "models").resolve())
    model_path, features_path = resolve_trained_model_paths(
        cwd=cwd,
        model_path_cfg=model_path_cfg,
        models_dir=models_dir,
        train_model_name=train_model_name,
    )
    return load_lightgbm_inference_model(
        model_path=model_path,
        features_path=features_path,
        required_bars=required_bars,
    )


class LightGBMInferenceModel(Model):
    """Обёртка над LGBMClassifier из train: предикт по последней строке окна фич."""

    def __init__(self, clf, feature_columns: list[str], *, required_bars: int, clip_bounds: dict[str, dict[str, float]] | None = None) -> None:
        self._clf = clf
        self._feature_columns = list(feature_columns)
        self._required_bars = required_bars
        self._clip_bounds = clip_bounds or {}

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def required_bars(self) -> int:
        return self._required_bars

    def predict(self, features: pd.DataFrame) -> dict:
        if features.empty:
            return {"score": 0.0, "p_long": 0.5, "p_short": 0.5}
        row = features.iloc[-1:].copy()
        missing = [c for c in self._feature_columns if c not in row.columns]
        if missing:
            raise ValueError(
                "В окне фич нет колонок модели "
                f"({len(missing)} шт., пример: {missing[:10]}). "
                "Сверь сборку фич с dataset pipeline и список feature_columns из обучения."
            )
        X = row[self._feature_columns].copy()
        if self._clip_bounds:
            for col, bounds in self._clip_bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=bounds["lower"], upper=bounds["upper"])
        if "symbol" in X.columns:
            X["symbol"] = X["symbol"].astype(str).astype("category")
        proba = self._clf.predict_proba(X)[0]
        classes = list(getattr(self._clf, "classes_", range(len(proba))))
        pmap = {int(c): float(p) for c, p in zip(classes, proba)}
        p_long = float(pmap.get(1, proba[-1] if len(proba) > 1 else 0.5))
        p_short = float(pmap.get(0, proba[0] if len(proba) > 1 else 1.0 - p_long))
        score = (p_long - 0.5) * 2.0
        return {"score": score, "p_long": p_long, "p_short": p_short}
