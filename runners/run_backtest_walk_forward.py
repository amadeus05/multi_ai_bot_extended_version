from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from application.backtest_engine import BacktestEngine
from application.inference.walk_forward_model import WalkForwardPredictionModel
from application.training.data_loader import load_training_frame
from application.training.walk_forward_pipeline import WalkForwardPipeline
from core.config.backtest_config import BacktestConfig
from core.config.train_config import TrainConfig


def _filter_backtest_frame(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if start:
        t0 = pd.to_datetime(start, errors="coerce")
        if pd.isna(t0):
            raise ValueError(f"BACKTEST_START не распарсился: {start!r}")
        out = out.loc[out["timestamp"] >= t0]
    if end:
        t1 = pd.to_datetime(end, errors="coerce")
        if pd.isna(t1):
            raise ValueError(f"BACKTEST_END не распарсился: {end!r}")
        out = out.loc[out["timestamp"] <= t1]
    return out.sort_values("timestamp").reset_index(drop=True)


async def main() -> None:
    backtest_cfg = BacktestConfig.from_env()
    train_cfg = TrainConfig.from_env()
    if backtest_cfg.backtest_strict_oos and not backtest_cfg.backtest_start:
        raise ValueError("BACKTEST_STRICT_OOS=1: для walk-forward тоже задайте BACKTEST_START.")

    # --- Load & filter dataset ---
    dataset = load_training_frame(backtest_cfg.dataset_dir, backtest_cfg.symbols, backtest_cfg.timeframe)
    if dataset.empty:
        raise RuntimeError("Пустой датасет для walk-forward backtest. Сначала запусти dataset pipeline.")
    dataset = _filter_backtest_frame(
        dataset,
        start=backtest_cfg.backtest_start,
        end=backtest_cfg.backtest_end,
    )

    # --- Walk-forward OOS predictions ---
    wf_pipeline = WalkForwardPipeline(train_cfg=train_cfg)
    wf_result = wf_pipeline.run(dataset)
    WalkForwardPipeline.save_artifacts(wf_result.predictions, wf_result.fold_details, train_cfg)

    # --- Build prediction model ---
    lookup = {
        (pd.Timestamp(row.timestamp), str(row.symbol)): (float(row.p_short), float(row.p_long))
        for row in wf_result.predictions.itertuples(index=False)
    }
    wf_model = WalkForwardPredictionModel(lookup=lookup, feature_columns=wf_result.feature_columns)

    # --- Load per-symbol data for backtest ---
    data_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in backtest_cfg.symbols:
        symbol_key = symbol.replace("/", "")
        dataset_path = (
            Path(backtest_cfg.dataset_dir) / f"symbol={symbol_key}" / f"timeframe={backtest_cfg.timeframe}" / "dataset.parquet"
        )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset for {symbol} not found: {dataset_path}. "
                "Run `python runners/run_dataset_pipeline.py` first."
            )
        frame = pd.read_parquet(dataset_path)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        frame = _filter_backtest_frame(frame, start=backtest_cfg.backtest_start, end=backtest_cfg.backtest_end)
        if frame.empty:
            raise ValueError(f"После фильтра BACKTEST_START/END не осталось строк для {symbol}.")
        data_by_symbol[symbol] = frame

    # --- Run backtest ---
    print(
        f"Walk-forward OOS mode: loaded {len(wf_result.predictions)} timestamp/symbol predictions "
        f"across {len(wf_result.fold_details)} folds."
    )
    from domain.strategy import BacktestParitySignalStrategy
    from domain.execution.exit_manager import ExitManager

    strategy = BacktestParitySignalStrategy(
        directional_proba_threshold=float(backtest_cfg.directional_proba_threshold),
        min_signal_gap=float(backtest_cfg.min_signal_gap),
        allow_longs=backtest_cfg.allow_longs,
        allow_shorts=backtest_cfg.allow_shorts,
    )
    exit_manager = ExitManager(slippage=float(backtest_cfg.slippage))

    engine = BacktestEngine(config=backtest_cfg, strategy=strategy, exit_manager=exit_manager)
    await engine.run(
        symbols=backtest_cfg.symbols,
        data_by_symbol=data_by_symbol,
        model=wf_model,
    )


if __name__ == "__main__":
    asyncio.run(main())
