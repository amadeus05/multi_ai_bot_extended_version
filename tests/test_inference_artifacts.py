from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.inference.artifacts import (
    load_feature_columns,
    load_feature_metadata,
    metadata_path_for_model,
    resolve_trained_model_paths,
)
from application.inference.lightgbm_inference_model import resolve_trained_model_paths as lightgbm_resolve_paths


def test_resolve_trained_model_paths_uses_existing_explicit_model(tmp_path: Path) -> None:
    model_path = tmp_path / "custom.joblib"
    model_path.write_bytes(b"model")

    resolved_model, resolved_metadata = resolve_trained_model_paths(
        cwd=tmp_path,
        model_path_cfg="custom.joblib",
        models_dir="models",
        train_model_name="lightgbm_target",
    )

    assert resolved_model == model_path
    assert resolved_metadata == tmp_path / "custom_features.json"


def test_resolve_trained_model_paths_falls_back_to_named_model(tmp_path: Path) -> None:
    resolved_model, resolved_metadata = resolve_trained_model_paths(
        cwd=tmp_path,
        model_path_cfg="missing.joblib",
        models_dir="models",
        train_model_name="lightgbm_target",
    )

    assert resolved_model == tmp_path / "models" / "lightgbm_target.joblib"
    assert resolved_metadata == tmp_path / "models" / "lightgbm_target_features.json"


def test_resolve_trained_model_paths_supports_non_lightgbm_suffix(tmp_path: Path) -> None:
    resolved_model, resolved_metadata = resolve_trained_model_paths(
        cwd=tmp_path,
        model_path_cfg="missing.pt",
        models_dir="models",
        train_model_name="lstm_candles_target",
        model_suffix=".pt",
        explicit_suffixes=(".pt",),
    )

    assert resolved_model == tmp_path / "models" / "lstm_candles_target.pt"
    assert resolved_metadata == tmp_path / "models" / "lstm_candles_target_features.json"


def test_load_feature_metadata_validates_feature_columns(tmp_path: Path) -> None:
    features_path = tmp_path / "model_features.json"
    features_path.write_text(json.dumps({"feature_columns": ["close", "volume"]}), encoding="utf-8")

    payload = load_feature_metadata(features_path)

    assert payload["feature_columns"] == ["close", "volume"]
    assert load_feature_columns(features_path) == ["close", "volume"]


def test_load_feature_metadata_rejects_missing_or_empty_feature_columns(tmp_path: Path) -> None:
    features_path = tmp_path / "model_features.json"
    features_path.write_text(json.dumps({"feature_columns": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_columns"):
        load_feature_metadata(features_path)


def test_metadata_path_for_model_and_lightgbm_reexport_are_compatible(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"

    assert metadata_path_for_model(model_path) == tmp_path / "model_features.json"
    assert lightgbm_resolve_paths is resolve_trained_model_paths
