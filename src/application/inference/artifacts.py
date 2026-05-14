from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_artifact_path(cwd: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else cwd / resolved


def metadata_path_for_model(model_path: Path, suffix: str = "_features.json") -> Path:
    return model_path.parent / f"{model_path.stem}{suffix}"


def resolve_trained_model_paths(
    *,
    cwd: Path,
    model_path_cfg: str,
    models_dir: str,
    train_model_name: str,
    model_suffix: str = ".joblib",
    explicit_suffixes: tuple[str, ...] = (".joblib", ".pkl", ".pickle"),
    metadata_suffix: str = "_features.json",
) -> tuple[Path, Path]:
    model_path_cfg_resolved = resolve_artifact_path(cwd, model_path_cfg)
    if model_path_cfg_resolved.exists() and model_path_cfg_resolved.suffix.lower() in explicit_suffixes:
        model_path = model_path_cfg_resolved
    else:
        models_dir_path = resolve_artifact_path(cwd, models_dir)
        model_path = models_dir_path / f"{train_model_name}{model_suffix}"
    return model_path, metadata_path_for_model(model_path, suffix=metadata_suffix)


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_metadata(features_path: Path) -> dict[str, Any]:
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Feature metadata not found: {features_path}. It should be saved next to the model artifact."
        )
    payload = read_json_file(features_path)
    feature_columns = payload.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Invalid {features_path}: expected non-empty feature_columns")
    return payload


def load_feature_columns(features_path: Path) -> list[str]:
    return list(load_feature_metadata(features_path)["feature_columns"])
