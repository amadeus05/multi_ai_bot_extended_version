from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardRunConfig:
    n_splits: int
    split_mode: str
    monthly_train_months: int
    monthly_test_months: int
    monthly_window_mode: str
    purge_gap: int = 0


@dataclass
class ModelingDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    unique_timestamps: np.ndarray | None = None
    prediction_frame: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    def resolved_unique_timestamps(self) -> np.ndarray:
        if self.unique_timestamps is not None:
            return self.unique_timestamps
        return np.sort(self.frame["timestamp"].dropna().unique())


@dataclass(frozen=True)
class FoldData:
    fold_idx: int
    original_train_timestamps: np.ndarray
    train_timestamps: np.ndarray
    test_timestamps: np.ndarray
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_columns: list[str]
    prediction_test_frame: pd.DataFrame | None = None


@dataclass
class FoldPrediction:
    predictions: pd.DataFrame
    fold_details: dict
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None


@dataclass
class WalkForwardRunResult:
    predictions: pd.DataFrame
    fold_details: list[dict]
    feature_columns: list[str]
    fold_predictions: list[FoldPrediction] = field(default_factory=list)


class WalkForwardAdapter(Protocol):
    model_id: str

    def prepare_dataset(self, dataset: pd.DataFrame) -> ModelingDataset:
        ...

    def should_skip_fold(self, fold: FoldData) -> bool:
        ...

    def fit_predict_fold(self, fold: FoldData) -> FoldPrediction:
        ...
