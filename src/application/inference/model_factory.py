from __future__ import annotations

from pathlib import Path

from core.config.base import BaseConfig
from core.config.settings import TradingSettings
from core.interfaces.model import Model

from application.inference.candle_lstm_inference_model import load_default_candle_lstm_model
from application.inference.lightgbm_inference_model import load_default_lightgbm_model


def load_inference_model(*, cwd: Path, trading: TradingSettings) -> Model:
    model_kind = BaseConfig.env_str("MODEL_KIND", "lightgbm").strip().lower()
    if model_kind in {"lightgbm", "lgbm"}:
        return load_default_lightgbm_model(
            cwd=cwd,
            model_path_cfg=trading.model_path,
            required_bars=int(BaseConfig.env_str("LIGHTGBM_REQUIRED_BARS", "250")),
            train_model_name=BaseConfig.env_str("LIGHTGBM_MODEL_NAME", "lightgbm_target"),
        )
    if model_kind in {"candle_lstm", "lstm_candles"}:
        return load_default_candle_lstm_model(
            cwd=cwd,
            model_path_cfg=trading.model_path,
            train_model_name=BaseConfig.env_str("CANDLE_LSTM_MODEL_NAME", "candle_lstm_target"),
        )
    raise ValueError("Unknown MODEL_KIND={!r}. Use 'lightgbm' or 'candle_lstm'.".format(model_kind))
