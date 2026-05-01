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
from application.inference.lightgbm_inference_model import (
    load_lightgbm_inference_model,
    resolve_trained_model_paths,
)
from core.config.backtest_config import BacktestConfig


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
    cfg = BacktestConfig.from_env()
    if cfg.backtest_strict_oos and not cfg.backtest_start:
        raise ValueError(
            "BACKTEST_STRICT_OOS=1: задайте BACKTEST_START (начало OOS-окна), "
            "иначе бэктест по полному parquet — in-sample относительно prod-модели "
            "(train_pipeline обучает финальную LGBM на всех строках датасета)."
        )
    cwd = Path.cwd()
    models_dir = str((cwd / "models").resolve())
    train_model_name = "lightgbm_target"
    model_path, features_path = resolve_trained_model_paths(
        cwd=cwd,
        model_path_cfg=cfg.model_path,
        models_dir=models_dir,
        train_model_name=train_model_name,
    )
    model = load_lightgbm_inference_model(
        model_path=model_path,
        features_path=features_path,
        required_bars=250,
    )

    data_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in cfg.symbols:
        symbol_key = symbol.replace("/", "")
        dataset_path = Path(cfg.dataset_dir) / f"symbol={symbol_key}" / f"timeframe={cfg.timeframe}" / "dataset.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset for {symbol} not found: {dataset_path}. "
                "Run `python runners/run_dataset_pipeline.py` first."
            )
        frame = pd.read_parquet(dataset_path)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        else:
            frame["symbol"] = frame["symbol"].fillna(symbol).astype(str)
        frame = _filter_backtest_frame(
            frame,
            start=cfg.backtest_start,
            end=cfg.backtest_end,
        )
        if frame.empty:
            raise ValueError(
                f"После фильтра BACKTEST_START/END не осталось строк для {symbol}. "
                "Проверь даты и parquet."
            )
        if cfg.backtest_start or cfg.backtest_end:
            tmin, tmax = frame["timestamp"].min(), frame["timestamp"].max()
            print(
                f"Окно бэктеста {symbol}: [{tmin} .. {tmax}] "
                f"(BACKTEST_START/END заданы)"
            )
        data_by_symbol[symbol] = frame

    from domain.strategy import BacktestParitySignalStrategy
    from domain.execution.exit_manager import ExitManager

    strategy = BacktestParitySignalStrategy(
        directional_proba_threshold=float(cfg.directional_proba_threshold),
        min_signal_gap=float(cfg.min_signal_gap),
        allow_longs=cfg.allow_longs,
        allow_shorts=cfg.allow_shorts,
    )
    exit_manager = ExitManager(slippage=float(cfg.slippage))

    engine = BacktestEngine(config=cfg, strategy=strategy, exit_manager=exit_manager)
    await engine.run(
        symbols=cfg.symbols,
        data_by_symbol=data_by_symbol,
        model=model,
    )


if __name__ == "__main__":
    asyncio.run(main())
