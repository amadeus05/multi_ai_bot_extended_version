from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from application.backtest_replay_reporter import BacktestReplayReporter
from application.backtest_trade_export import export_closed_trades_csv
from application.inference.walk_forward_model import WalkForwardPredictionModel
from application.training.data_loader import load_training_frame
from application.training.walk_forward_pipeline import WalkForwardPipeline
from application.trading_engine import TradingEngine
from core.config.backtest_config import BacktestConfig
from core.config.train_config import TrainConfig
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.ml.labeling.barrier_policy import BarrierPolicy
from domain.ml.labeling.models import LabelingConfig
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.adaptive_risk_manager import AdaptiveRiskManager
from domain.risk.models.risk_profile import RiskProfile
from domain.strategy import BacktestParitySignalStrategy
from infrastructure.data_providers.historical_replay_provider import HistoricalReplayProvider
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


def _filter_backtest_frame(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if start:
        t0 = pd.to_datetime(start, errors="coerce")
        if pd.isna(t0):
            raise ValueError(f"BACKTEST_START did not parse: {start!r}")
        out = out.loc[out["timestamp"] >= t0]
    if end:
        t1 = pd.to_datetime(end, errors="coerce")
        if pd.isna(t1):
            raise ValueError(f"BACKTEST_END did not parse: {end!r}")
        out = out.loc[out["timestamp"] <= t1]
    return out.sort_values("timestamp").reset_index(drop=True)


def _load_symbol_frame(cfg: BacktestConfig, symbol: str) -> pd.DataFrame:
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
    frame = _filter_backtest_frame(frame, start=cfg.backtest_start, end=cfg.backtest_end)
    if frame.empty:
        raise ValueError(f"No rows remain for {symbol} after BACKTEST_START/END filtering.")
    return frame


async def main() -> None:
    backtest_cfg = BacktestConfig.from_env()
    train_cfg = TrainConfig.from_env()
    if len(backtest_cfg.symbols) != 1:
        raise ValueError(
            "TradingEngine walk-forward backtest currently supports exactly one symbol. "
            "Set SYMBOLS to one instrument, for example SYMBOLS=BTC/USDT."
        )
    if backtest_cfg.backtest_strict_oos and not backtest_cfg.backtest_start:
        raise ValueError("BACKTEST_STRICT_OOS=1 requires BACKTEST_START.")

    symbol = backtest_cfg.symbols[0]
    dataset = load_training_frame(backtest_cfg.dataset_dir, backtest_cfg.symbols, backtest_cfg.timeframe)
    if dataset.empty:
        raise RuntimeError("Empty dataset for walk-forward backtest. Run dataset pipeline first.")
    dataset = _filter_backtest_frame(dataset, start=backtest_cfg.backtest_start, end=backtest_cfg.backtest_end)
    if dataset.empty:
        raise RuntimeError("Dataset is empty after BACKTEST_START/END filtering.")

    wf_pipeline = WalkForwardPipeline(train_cfg=train_cfg)
    wf_result = wf_pipeline.run(dataset)
    WalkForwardPipeline.save_artifacts(wf_result.predictions, wf_result.fold_details, train_cfg)
    lookup = {
        (pd.Timestamp(row.timestamp), str(row.symbol)): (float(row.p_short), float(row.p_long))
        for row in wf_result.predictions.itertuples(index=False)
    }
    wf_model = WalkForwardPredictionModel(lookup=lookup, feature_columns=wf_result.feature_columns)

    frame = _load_symbol_frame(backtest_cfg, symbol)
    strict_predictions = os.getenv("WF_REPLAY_STRICT_PREDICTIONS", "0").strip().lower() in ("1", "true", "yes")
    start_bar_idx = max(0, int(backtest_cfg.backtest_skip_initial_bars))
    if strict_predictions:
        symbol_predictions = wf_result.predictions.loc[wf_result.predictions["symbol"].astype(str) == symbol]
        if symbol_predictions.empty:
            raise RuntimeError(f"Walk-forward produced no OOS predictions for {symbol}.")
        first_pred_ts = pd.Timestamp(symbol_predictions["timestamp"].min())
        first_pred_matches = frame.index[pd.to_datetime(frame["timestamp"], errors="coerce") >= first_pred_ts].tolist()
        if not first_pred_matches:
            raise RuntimeError(f"No replay rows at or after first OOS prediction timestamp {first_pred_ts}.")
        start_bar_idx = max(start_bar_idx, int(first_pred_matches[0]))

    exchange = SimulatedExchange(
        commission=float(backtest_cfg.taker_com),
        slippage=float(backtest_cfg.slippage),
        leverage=float(backtest_cfg.leverage),
    )
    labeling_cfg = LabelingConfig.from_env()
    barrier_policy = BarrierPolicy(labeling_cfg)
    portfolio = PortfolioManager(cash={"USDT": float(backtest_cfg.initial_capital)})
    data_provider = HistoricalReplayProvider(symbol, frame, skip_initial_bars=start_bar_idx)
    strategy = BacktestParitySignalStrategy(
        directional_proba_threshold=float(backtest_cfg.directional_proba_threshold),
        min_signal_gap=float(backtest_cfg.min_signal_gap),
        allow_longs=backtest_cfg.allow_longs,
        allow_shorts=backtest_cfg.allow_shorts,
    )
    engine = TradingEngine(
        exchange=exchange,
        data_provider=data_provider,
        model=wf_model,
        strategy=strategy,
        risk_manager=AdaptiveRiskManager(profile=RiskProfile.from_env()),
        portfolio=portfolio,
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=barrier_policy.slippage),
        barrier_policy=barrier_policy,
    )
    reporter = BacktestReplayReporter(
        config=backtest_cfg,
        portfolio=portfolio,
        charts_dir=backtest_cfg.backtest_charts_dir,
    )
    engine.set_execution_listener(reporter.on_execution)

    print(
        f"TradingEngine Walk-forward OOS backtest | {symbol} | rows={len(frame)} | "
        f"predictions={len(wf_result.predictions)} | folds={len(wf_result.fold_details)} | "
        f"first_bar_index={start_bar_idx} | strict_predictions={int(strict_predictions)} | "
        f"capital={backtest_cfg.initial_capital}"
    )
    print("-" * 80)

    for tick in data_provider.iter_replay_ticks():
        await engine.on_market_event(tick)
        mark = float(tick.close if tick.close is not None else tick.price)
        reporter.record_equity(tick.ts, {symbol: mark})

    final_ts = data_provider.last_timestamp
    final_close = data_provider.last_close
    if final_ts is not None and final_close is not None:
        await engine.close_all_at_market({symbol: float(final_close)}, final_ts)
        reporter.record_equity(final_ts, {symbol: float(final_close)})

    snap = portfolio.get_state_snapshot()
    print(
        f"Done | trades={snap['trades_count']} | closed={snap['closed_trades_count']} | "
        f"realized_pnl={snap['realized_pnl']:.4f}"
    )
    report = reporter.finish()
    export_closed_trades_csv(
        report,
        Path(backtest_cfg.backtest_charts_dir) / "wf_trading_engine_closed_trades.csv",
        source="trading_engine",
    )


if __name__ == "__main__":
    asyncio.run(main())
