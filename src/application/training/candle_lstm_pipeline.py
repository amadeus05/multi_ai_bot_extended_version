from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from application.training.candle_lstm_features import CANDLE_LSTM_FEATURE_COLUMNS, build_candle_lstm_feature_frame
from application.training.data_loader import load_training_frame
from application.training.lstm_model import LSTMClassifier
from application.training.metrics import evaluate_model
from application.training.sequence_dataset import SequenceDataset, SequenceStandardizer, build_history_by_symbol
from application.training.splits import build_timestamp_splits
from core.config.base import BaseConfig
from core.config.train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleLSTMConfig:
    sequence_length: int = 64
    batch_size: int = 128
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    early_stopping_patience: int = 8
    min_epochs_before_early_stop: int = 10
    early_stopping_min_delta: float = 5e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    min_train_rows: int = 200
    model_name: str = "candle_lstm_target"

    @classmethod
    def from_env(cls) -> "CandleLSTMConfig":
        return cls(
            sequence_length=int(BaseConfig.env_str("CANDLE_LSTM_SEQUENCE_LENGTH", "64")),
            batch_size=int(BaseConfig.env_str("CANDLE_LSTM_BATCH_SIZE", "128")),
            hidden_size=int(BaseConfig.env_str("CANDLE_LSTM_HIDDEN_SIZE", "64")),
            num_layers=int(BaseConfig.env_str("CANDLE_LSTM_NUM_LAYERS", "1")),
            dropout=float(BaseConfig.env_str("CANDLE_LSTM_DROPOUT", "0.1")),
            learning_rate=float(BaseConfig.env_str("CANDLE_LSTM_LR", "5e-4")),
            weight_decay=float(BaseConfig.env_str("CANDLE_LSTM_WEIGHT_DECAY", "1e-5")),
            epochs=int(BaseConfig.env_str("CANDLE_LSTM_EPOCHS", "30")),
            early_stopping_patience=int(BaseConfig.env_str("CANDLE_LSTM_EARLY_STOPPING_PATIENCE", "8")),
            min_epochs_before_early_stop=int(BaseConfig.env_str("CANDLE_LSTM_MIN_EPOCHS_BEFORE_EARLY_STOP", "10")),
            early_stopping_min_delta=float(BaseConfig.env_str("CANDLE_LSTM_EARLY_STOPPING_MIN_DELTA", "5e-4")),
            gradient_clip=float(BaseConfig.env_str("CANDLE_LSTM_GRADIENT_CLIP", "1.0")),
            validation_fraction=float(BaseConfig.env_str("CANDLE_LSTM_VALIDATION_FRACTION", "0.15")),
            min_train_rows=int(BaseConfig.env_str("CANDLE_LSTM_MIN_TRAIN_ROWS", "200")),
            model_name=BaseConfig.env_str("CANDLE_LSTM_MODEL_NAME", "candle_lstm_target"),
        )


@dataclass
class CandleLSTMTrainingResult:
    oos_metrics: dict
    fold_details: list[dict]
    oos_predictions: pd.DataFrame
    feature_columns: list[str]
    artifact_paths: dict[str, str]


class CandleLSTMTrainPipeline:
    def __init__(self, train_cfg: TrainConfig, lstm_cfg: CandleLSTMConfig | None = None) -> None:
        self.train_cfg = train_cfg
        self.lstm_cfg = lstm_cfg or CandleLSTMConfig.from_env()

    def run(self) -> CandleLSTMTrainingResult:
        set_seed(self.train_cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Candle LSTM training device=%s", device)

        raw_dataset = load_training_frame(self.train_cfg.dataset_dir, self.train_cfg.symbols, self.train_cfg.timeframe)
        if raw_dataset.empty:
            raise RuntimeError("Training dataset is empty. Run dataset pipeline first.")
        sample_frame, history_by_symbol, feature_columns = prepare_candle_lstm_frames(raw_dataset)
        if sample_frame.empty:
            raise RuntimeError("No directional samples for Candle LSTM training.")

        metrics, fold_details, predictions = self._run_walk_forward(
            sample_frame=sample_frame,
            history_by_symbol=history_by_symbol,
            feature_columns=feature_columns,
            device=device,
        )
        prod_model, standardizer, prod_meta = self._train_production_model(
            sample_frame=sample_frame,
            history_by_symbol=history_by_symbol,
            feature_columns=feature_columns,
            device=device,
        )
        artifact_paths = self._save_artifacts(
            model=prod_model,
            standardizer=standardizer,
            feature_columns=feature_columns,
            metrics=metrics,
            fold_details=fold_details,
            predictions=predictions,
            production_meta=prod_meta,
        )
        return CandleLSTMTrainingResult(
            oos_metrics=metrics,
            fold_details=fold_details,
            oos_predictions=predictions,
            feature_columns=feature_columns,
            artifact_paths=artifact_paths,
        )

    def _run_walk_forward(
        self,
        *,
        sample_frame: pd.DataFrame,
        history_by_symbol: dict[str, pd.DataFrame],
        feature_columns: list[str],
        device: torch.device,
    ) -> tuple[dict, list[dict], pd.DataFrame]:
        unique_ts = np.sort(sample_frame["timestamp"].unique())
        splits = build_timestamp_splits(
            unique_ts=unique_ts,
            n_splits=self.train_cfg.n_splits,
            split_mode=self.train_cfg.split_mode,
            monthly_train_months=self.train_cfg.monthly_train_months,
            monthly_test_months=self.train_cfg.monthly_test_months,
            monthly_window_mode=self.train_cfg.monthly_window_mode,
        )
        if not splits:
            raise RuntimeError("No valid walk-forward splits for Candle LSTM.")

        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_y_proba: list[np.ndarray] = []
        fold_details: list[dict] = []
        prediction_frames: list[pd.DataFrame] = []

        for fold_idx, original_train_ts, test_ts in splits:
            train_ts = original_train_ts
            if self.train_cfg.purge_gap > 0 and len(train_ts) > self.train_cfg.purge_gap:
                train_ts = train_ts[: -self.train_cfg.purge_gap]
            train_df = sample_frame.loc[sample_frame["timestamp"].isin(set(train_ts))].copy()
            test_df = sample_frame.loc[sample_frame["timestamp"].isin(set(test_ts))].copy()
            if train_df.empty or test_df.empty or len(sorted(train_df["Target"].unique().tolist())) < 2:
                continue

            raw_train_dataset = SequenceDataset(
                train_df,
                history_by_symbol,
                feature_columns,
                self.lstm_cfg.sequence_length,
            )
            if len(raw_train_dataset) < self.lstm_cfg.min_train_rows:
                logger.warning("Candle LSTM fold %s skipped: only %s train sequences.", fold_idx, len(raw_train_dataset))
                continue
            train_indices, eval_indices = split_train_eval_indices(
                raw_train_dataset,
                self.lstm_cfg.validation_fraction,
                purge_gap_timestamps=max(self.lstm_cfg.sequence_length - 1, self.train_cfg.purge_gap),
            )
            if not train_indices or not eval_indices:
                logger.warning("Candle LSTM fold %s skipped: empty train/eval after internal purge.", fold_idx)
                continue

            standardizer = SequenceStandardizer().fit(raw_train_dataset.sequences_array(train_indices))
            train_dataset = SequenceDataset(
                train_df,
                history_by_symbol,
                feature_columns,
                self.lstm_cfg.sequence_length,
                standardizer=standardizer,
            )
            test_dataset = SequenceDataset(
                test_df,
                history_by_symbol,
                feature_columns,
                self.lstm_cfg.sequence_length,
                standardizer=standardizer,
            )
            if len(test_dataset) == 0:
                continue

            model, best_epoch, best_eval_loss = train_lstm_model(
                train_dataset,
                train_indices=train_indices,
                eval_indices=eval_indices,
                cfg=self.lstm_cfg,
                device=device,
            )
            y_proba, y_pred = predict_dataset(model, test_dataset, self.lstm_cfg.batch_size, device)
            meta = test_dataset.metadata_frame()
            y_true = meta["Target"].to_numpy(dtype=int)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": meta["timestamp"].values,
                        "symbol": meta["symbol"].values,
                        "Target": y_true,
                        "prediction": y_pred,
                        "p_short": y_proba[:, 0],
                        "p_long": y_proba[:, 1],
                        "fold": int(fold_idx),
                    }
                )
            )
            fold_auc = None
            if len(set(y_true.tolist())) > 1:
                fold_auc = float(evaluate_model(y_true, y_pred, y_proba, self.train_cfg.confidence_threshold)["roc_auc"])
            fold_details.append(
                {
                    "fold": int(fold_idx),
                    "split_mode": self.train_cfg.split_mode,
                    "train_sequences": int(len(train_dataset)),
                    "test_sequences": int(len(test_dataset)),
                    "best_epoch": int(best_epoch),
                    "best_eval_loss": float(best_eval_loss),
                    "accuracy": float((y_pred == y_true).mean()),
                    "roc_auc": fold_auc,
                    "purged_timestamps": int(self.train_cfg.purge_gap),
                    "timestamp_boundaries": {
                        "train_start": str(train_ts[0]) if len(train_ts) else None,
                        "train_end_after_purge": str(train_ts[-1]) if len(train_ts) else None,
                        "test_start": str(test_ts[0]) if len(test_ts) else None,
                        "test_end": str(test_ts[-1]) if len(test_ts) else None,
                    },
                }
            )
            logger.info(
                "Candle LSTM fold %s | train_seq=%s test_seq=%s best_epoch=%s acc=%.4f auc=%s",
                fold_idx,
                len(train_dataset),
                len(test_dataset),
                best_epoch,
                float((y_pred == y_true).mean()),
                f"{fold_auc:.4f}" if fold_auc is not None else "-",
            )

        if not all_y_true:
            raise RuntimeError("All Candle LSTM folds were skipped; cannot compute OOS metrics.")
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        y_proba_all = np.vstack(all_y_proba)
        metrics = evaluate_model(
            y_true_all,
            y_pred_all,
            y_proba_all,
            confidence_threshold=self.train_cfg.confidence_threshold,
        )
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
        predictions = predictions.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return metrics, fold_details, predictions

    def _train_production_model(
        self,
        *,
        sample_frame: pd.DataFrame,
        history_by_symbol: dict[str, pd.DataFrame],
        feature_columns: list[str],
        device: torch.device,
    ) -> tuple[LSTMClassifier, SequenceStandardizer, dict]:
        raw_dataset = SequenceDataset(sample_frame, history_by_symbol, feature_columns, self.lstm_cfg.sequence_length)
        if len(raw_dataset) < self.lstm_cfg.min_train_rows:
            raise RuntimeError(f"Only {len(raw_dataset)} Candle LSTM sequences, need at least {self.lstm_cfg.min_train_rows}.")
        train_indices, eval_indices = split_train_eval_indices(
            raw_dataset,
            self.lstm_cfg.validation_fraction,
            purge_gap_timestamps=max(self.lstm_cfg.sequence_length - 1, self.train_cfg.purge_gap),
        )
        if not train_indices:
            raise RuntimeError("Candle LSTM production split produced no train sequences.")
        standardizer = SequenceStandardizer().fit(raw_dataset.sequences_array(train_indices))
        dataset = SequenceDataset(
            sample_frame,
            history_by_symbol,
            feature_columns,
            self.lstm_cfg.sequence_length,
            standardizer=standardizer,
        )
        model, best_epoch, best_eval_loss = train_lstm_model(
            dataset,
            train_indices=train_indices,
            eval_indices=eval_indices,
            cfg=self.lstm_cfg,
            device=device,
        )
        return model, standardizer, {
            "rows": int(len(dataset)),
            "train_sequences": int(len(train_indices)),
            "eval_sequences": int(len(eval_indices)),
            "best_epoch": int(best_epoch),
            "best_eval_loss": float(best_eval_loss),
        }

    def _save_artifacts(
        self,
        *,
        model: LSTMClassifier,
        standardizer: SequenceStandardizer,
        feature_columns: list[str],
        metrics: dict,
        fold_details: list[dict],
        predictions: pd.DataFrame,
        production_meta: dict,
    ) -> dict[str, str]:
        models_dir = self.train_cfg.models_dir_path
        models_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.lstm_cfg.model_name
        model_path = models_dir / f"{model_name}.pt"
        metrics_path = models_dir / f"{model_name}_metrics.json"
        features_path = models_dir / f"{model_name}_features.json"
        history_path = models_dir / f"{model_name}_train_history.json"
        predictions_path = models_dir / f"{model_name}_walk_forward_oos_predictions.csv"

        payload = {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "standardizer": standardizer.to_payload(),
            "sequence_length": int(self.lstm_cfg.sequence_length),
            "symbols": list(self.train_cfg.symbols),
            "model_args": model_args_payload(feature_columns, self.lstm_cfg),
            "model_kind": "candle_lstm",
        }
        torch.save(payload, model_path)
        predictions.to_csv(predictions_path, index=False)
        metrics_payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_kind": "candle_lstm",
            "model_name": model_name,
            "oos_metrics": metrics,
            "fold_details": fold_details,
            "production": production_meta,
            "feature_count": int(len(feature_columns)),
            "sequence_length": int(self.lstm_cfg.sequence_length),
            "symbols": list(self.train_cfg.symbols),
            "confidence_threshold": float(self.train_cfg.confidence_threshold),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")
        features_path.write_text(
            json.dumps(
                {
                    "model_kind": "candle_lstm",
                    "feature_columns": feature_columns,
                    "sequence_length": int(self.lstm_cfg.sequence_length),
                    "symbols": list(self.train_cfg.symbols),
                    "model_args": model_args_payload(feature_columns, self.lstm_cfg),
                    "label_mapping": {"short": 0, "long": 1},
                    "inverse_label_mapping": {"0": -1, "1": 1},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history = []
        if history_path.exists():
            try:
                loaded = json.loads(history_path.read_text(encoding="utf-8"))
                history = loaded if isinstance(loaded, list) else []
            except json.JSONDecodeError:
                history = []
        history.append(
            {
                "run_timestamp_utc": metrics_payload["run_timestamp_utc"],
                "model_name": model_name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "mcc": metrics.get("mcc"),
                "sequence_length": int(self.lstm_cfg.sequence_length),
                "feature_count": int(len(feature_columns)),
                **production_meta,
            }
        )
        history_path.write_text(json.dumps(history[-200:], indent=2, default=str), encoding="utf-8")
        return {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "features": str(features_path),
            "walk_forward_predictions": str(predictions_path),
            "train_history": str(history_path),
        }


def prepare_candle_lstm_frames(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    full_frame = dataset.copy()
    full_frame["timestamp"] = pd.to_datetime(full_frame["timestamp"], errors="coerce")
    full_frame = full_frame.dropna(subset=["timestamp", "symbol", "Target"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    feature_frame = build_candle_lstm_feature_frame(full_frame)

    raw_labels = full_frame["Target"].astype(int)
    sample_frame = full_frame.loc[raw_labels != 0, ["timestamp", "symbol", "Target"]].copy()
    sample_frame["Target"] = sample_frame["Target"].astype(int).map({-1: 0, 1: 1})
    sample_frame = sample_frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    history_by_symbol = build_history_by_symbol(feature_frame, CANDLE_LSTM_FEATURE_COLUMNS)
    return sample_frame, history_by_symbol, list(CANDLE_LSTM_FEATURE_COLUMNS)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_eval_indices(
    dataset: SequenceDataset,
    validation_fraction: float,
    *,
    purge_gap_timestamps: int,
) -> tuple[list[int], list[int]]:
    n_items = len(dataset)
    if n_items < 2:
        return list(range(n_items)), []
    sample_timestamps = np.asarray([pd.Timestamp(sample.timestamp) for sample in dataset.samples])
    unique_timestamps = np.unique(sample_timestamps)
    if len(unique_timestamps) < 2:
        return list(range(n_items)), []
    eval_size = max(1, int(len(unique_timestamps) * validation_fraction))
    if eval_size >= len(unique_timestamps):
        eval_size = 1
    train_timestamps = unique_timestamps[:-eval_size]
    eval_timestamps = unique_timestamps[-eval_size:]
    if purge_gap_timestamps > 0 and len(train_timestamps):
        purge_count = min(int(purge_gap_timestamps), len(train_timestamps))
        train_timestamps = train_timestamps[:-purge_count] if purge_count else train_timestamps
    if len(train_timestamps) == 0:
        return [], []
    train_ts_set = set(train_timestamps.tolist())
    eval_ts_set = set(eval_timestamps.tolist())
    train_indices = [idx for idx, ts in enumerate(sample_timestamps) if ts in train_ts_set]
    eval_indices = [idx for idx, ts in enumerate(sample_timestamps) if ts in eval_ts_set]
    return train_indices, eval_indices


def train_lstm_model(
    dataset: SequenceDataset,
    *,
    train_indices: list[int],
    eval_indices: list[int],
    cfg: CandleLSTMConfig,
    device: torch.device,
) -> tuple[LSTMClassifier, int, float]:
    fit_loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_indices or train_indices), batch_size=cfg.batch_size, shuffle=False)
    model = LSTMClassifier(
        input_size=len(dataset.feature_columns),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state = None
    best_eval_loss = float("inf")
    stale_epochs = 0
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, fit_loader, criterion, optimizer, device, cfg, train_mode=True)
        eval_loss = run_epoch(model, eval_loader, criterion, optimizer, device, cfg, train_mode=False)
        if eval_loss < (best_eval_loss - cfg.early_stopping_min_delta):
            best_eval_loss = eval_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        logger.info("Candle LSTM epoch=%s train_loss=%.5f eval_loss=%.5f", epoch, train_loss, eval_loss)
        if epoch >= cfg.min_epochs_before_early_stop and stale_epochs >= cfg.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_eval_loss


def run_epoch(
    model: LSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: CandleLSTMConfig,
    *,
    train_mode: bool,
) -> float:
    model.train(train_mode)
    total_loss = 0.0
    total_rows = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / max(total_rows, 1)


def predict_dataset(
    model: LSTMClassifier,
    dataset: SequenceDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probas: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            probas.append(proba)
            preds.append(np.argmax(proba, axis=1))
    if not probas:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(probas), np.concatenate(preds)


def model_args_payload(feature_columns: list[str], cfg: CandleLSTMConfig) -> dict:
    return {
        "input_size": int(len(feature_columns)),
        "hidden_size": int(cfg.hidden_size),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
    }
