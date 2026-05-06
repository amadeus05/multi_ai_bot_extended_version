from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from application.training.candle_lstm_features import build_candle_lstm_feature_frame
from application.training.lstm_model import LSTMClassifier
from application.training.sequence_dataset import SequenceStandardizer
from core.interfaces.model import Model


def load_candle_lstm_inference_model(*, model_path: Path) -> Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Candle LSTM model not found: {model_path}. Train it with runners/run_train_candle_lstm.py.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(model_path, map_location=device, weights_only=False)
    feature_columns = payload.get("feature_columns")
    model_args = payload.get("model_args") or {}
    standardizer_payload = payload.get("standardizer") or {}
    sequence_length = int(payload.get("sequence_length") or 0)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Invalid Candle LSTM payload in {model_path}: missing feature_columns.")
    if sequence_length <= 0:
        raise ValueError(f"Invalid Candle LSTM payload in {model_path}: missing sequence_length.")
    model = LSTMClassifier(
        input_size=int(model_args.get("input_size", len(feature_columns))),
        hidden_size=int(model_args.get("hidden_size", 64)),
        num_layers=int(model_args.get("num_layers", 1)),
        dropout=float(model_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    standardizer = SequenceStandardizer.from_payload(standardizer_payload)
    return CandleLSTMInferenceModel(
        model=model,
        feature_columns=feature_columns,
        standardizer=standardizer,
        sequence_length=sequence_length,
        device=device,
    )


def load_default_candle_lstm_model(
    *,
    cwd: Path,
    model_path_cfg: str,
    train_model_name: str = "candle_lstm_target",
) -> Model:
    model_path = Path(model_path_cfg)
    if not model_path.is_absolute():
        model_path = cwd / model_path
    if not model_path.exists() or model_path.suffix.lower() not in (".pt", ".pth"):
        model_path = cwd / "models" / f"{train_model_name}.pt"
    return load_candle_lstm_inference_model(model_path=model_path)


class CandleLSTMInferenceModel(Model):
    def __init__(
        self,
        *,
        model: LSTMClassifier,
        feature_columns: list[str],
        standardizer: SequenceStandardizer,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        self._model = model
        self._feature_columns = list(feature_columns)
        self._standardizer = standardizer
        self._sequence_length = int(sequence_length)
        self._device = device

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def required_bars(self) -> int:
        return max(self._sequence_length, 120)

    def predict(self, features: pd.DataFrame) -> dict:
        if features.empty:
            return {"score": 0.0, "p_long": 0.5, "p_short": 0.5}
        candle_features = build_candle_lstm_feature_frame(features)
        if candle_features.empty or len(candle_features) < self._sequence_length:
            return {"score": 0.0, "p_long": 0.5, "p_short": 0.5}
        missing = [column for column in self._feature_columns if column not in candle_features.columns]
        if missing:
            raise ValueError(f"Candle LSTM feature frame is missing model columns: {missing[:10]}")
        window = candle_features[self._feature_columns].tail(self._sequence_length).to_numpy(dtype=np.float32, copy=True)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        sequence = self._standardizer.transform(window[None, :, :])
        with torch.no_grad():
            tensor = torch.from_numpy(sequence).to(self._device)
            proba = torch.softmax(self._model(tensor), dim=1).cpu().numpy()[0]
        p_short = float(proba[0])
        p_long = float(proba[1])
        score = (p_long - 0.5) * 2.0
        return {"score": score, "p_long": p_long, "p_short": p_short}
