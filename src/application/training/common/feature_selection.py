from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Target"
TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
RESERVED_COLUMNS = {TARGET_COLUMN, TIMESTAMP_COLUMN, "barrier_stop_pct", "barrier_take_pct"}
EXCLUDED_RAW_FEATURE_COLUMNS = {"open", "high", "low", "close", "volume"}


def resolve_train_feature_allowlist(subset: str) -> list[str] | None:
    """Имена колонок для обучения; None = без ограничения (все числовые из датасета)."""
    key = (subset or "").strip()
    if not key:
        return None
    if key == "legacy_mvp_v1":
        from domain.ml.features.config import LEGACY_MVP_V1_FEATURES

        return list(LEGACY_MVP_V1_FEATURES)
    raise ValueError(
        f"Неизвестный TRAIN_FEATURE_SUBSET={subset!r}. Используйте пустую строку или 'legacy_mvp_v1'."
    )


def ensure_legacy_allowlist_columns(dataset: pd.DataFrame, allowlist: list[str]) -> None:
    """
    Старый parquet мог быть собран до rsi_1h / mcc_sign_agreement_btc_24h.
    Дописываем колонки in-place, чтобы не падать и совпадать с легаси-списком из 21 фичи.
    """
    from domain.ml.features.indicators import compute_rsi

    if "mcc_sign_agreement_btc_24h" in allowlist and "mcc_sign_agreement_btc_24h" not in dataset.columns:
        dataset["mcc_sign_agreement_btc_24h"] = 0.0
        logger.info(
            "Дополнена колонка mcc_sign_agreement_btc_24h=0 (в parquet её не было — старый сборка датасета)."
        )
    if "rsi_1h" in allowlist and "rsi_1h" not in dataset.columns:
        if "close" not in dataset.columns:
            raise RuntimeError(
                "В датасете нет колонки close — нельзя восстановить rsi_1h. "
                "Запусти: python runners/run_dataset_pipeline.py"
            )
        close = pd.to_numeric(dataset["close"], errors="coerce")
        dataset["rsi_1h"] = compute_rsi(close, length=14)
        logger.info(
            "Вычислена rsi_1h из close (в parquet колонки не было — пересборка датасета всё равно желательна)."
        )


def filter_feature_columns_to_allowlist(
    dataset: pd.DataFrame,
    allowlist: list[str],
) -> list[str]:
    """Оставляет только колонки из allowlist, в том же порядке; проверяет наличие в frame."""
    ensure_legacy_allowlist_columns(dataset, allowlist)
    missing = [name for name in allowlist if name not in dataset.columns]
    if missing:
        raise RuntimeError(
            "В датасете нет колонок, нужных для обучения: "
            + ", ".join(missing[:25])
            + (f" … (+{len(missing) - 25} ещё)" if len(missing) > 25 else "")
            + ". Пересобери датасет: python runners/run_dataset_pipeline.py"
        )
    return list(allowlist)


def select_feature_columns(dataset: pd.DataFrame, use_symbol_feature: bool) -> list[str]:
    columns: list[str] = []
    for column in dataset.columns:
        if column in RESERVED_COLUMNS or column in EXCLUDED_RAW_FEATURE_COLUMNS:
            continue
        if column == SYMBOL_COLUMN:
            if use_symbol_feature:
                columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            columns.append(column)
    if not columns:
        raise RuntimeError("No usable feature columns found.")
    return columns
