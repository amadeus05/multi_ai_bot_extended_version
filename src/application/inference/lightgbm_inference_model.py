from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from core.interfaces.model import Model


def resolve_trained_model_paths(
    *,
    cwd: Path,
    model_path_cfg: str,
    models_dir: str,
    train_model_name: str,
) -> tuple[Path, Path]:
    """
    Находит артефакты обучения: ``{name}.joblib`` и ``{name}_features.json``.
    Если ``MODEL_PATH`` указывает на существующий .joblib/.pkl — берём его и json по stem.
    Иначе — ``{MODELS_DIR}/{TRAIN_MODEL_NAME}.joblib``.
    """
    mp = Path(model_path_cfg)
    if not mp.is_absolute():
        mp = cwd / mp
    if mp.exists() and mp.suffix.lower() in (".joblib", ".pkl", ".pickle"):
        model_path = mp
    else:
        md = Path(models_dir)
        if not md.is_absolute():
            md = cwd / md
        model_path = md / f"{train_model_name}.joblib"
    features_path = model_path.parent / f"{model_path.stem}_features.json"
    return model_path, features_path


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
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Список фич не найден: {features_path}. Должен лежать рядом с joblib после train."
        )
    payload = json.loads(features_path.read_text(encoding="utf-8"))
    feature_columns = payload.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Некорректный {features_path}: ожидался непустой feature_columns")
    clf = joblib.load(model_path)
    clip_bounds_path = model_path.parent / f"{model_path.stem}_clip_bounds.json"
    clip_bounds: dict[str, dict[str, float]] | None = None
    if clip_bounds_path.is_file():
        clip_bounds = json.loads(clip_bounds_path.read_text(encoding="utf-8"))
    return LightGBMInferenceModel(clf, feature_columns, required_bars=required_bars, clip_bounds=clip_bounds)


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
