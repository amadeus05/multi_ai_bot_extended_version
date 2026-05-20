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
from application.backtest_position_closer import close_all_positions_at_market
from application.backtest_trade_export import export_closed_trades_csv
from application.inference.walk_forward_model import WalkForwardPredictionModel
from application.training.data_loader import load_training_frame
from application.training.walk_forward_pipeline import WalkForwardPipeline
from application.trading_runtime_factory import build_trading_engine
from core.config.loader import load_backtest_settings
from core.config.train_config import TrainConfig
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.data_providers.historical_replay_provider import HistoricalMultiSymbolReplayProvider
from infrastructure.notifications import SilentNotifier
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


def _filter_backtest_frame(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "decision_time" not in out.columns:
        out["decision_time"] = out["timestamp"]
    out["decision_time"] = pd.to_datetime(out["decision_time"], errors="coerce")
    out["decision_time"] = out["decision_time"].fillna(out["timestamp"])
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


def _prediction_time(row) -> pd.Timestamp:
    value = getattr(row, "decision_time", None)
    if value is None or pd.isna(value):
        value = row.timestamp
    return pd.Timestamp(value)


def _load_symbol_frame(settings, symbol: str) -> pd.DataFrame:
    symbol_key = symbol.replace("/", "")
    dataset_path = (
        Path(settings.dataset_dir)
        / f"symbol={symbol_key}"
        / f"timeframe={settings.trading.timeframe}"
        / "dataset.parquet"
    )
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
    frame = _filter_backtest_frame(frame, start=settings.start, end=settings.end)
    if frame.empty:
        raise ValueError(f"No rows remain for {symbol} after BACKTEST_START/END filtering.")
    return frame


async def main() -> None:
    backtest_settings = load_backtest_settings()
    train_cfg = TrainConfig.from_env()
    trading = backtest_settings.trading
    symbols = list(trading.symbols)
    if backtest_settings.strict_oos and not backtest_settings.start:
        raise ValueError("BACKTEST_STRICT_OOS=1 requires BACKTEST_START.")

    dataset = load_training_frame(backtest_settings.dataset_dir, symbols, trading.timeframe)
    if dataset.empty:
        raise RuntimeError("Empty dataset for walk-forward backtest. Run dataset pipeline first.")
    dataset = _filter_backtest_frame(dataset, start=backtest_settings.start, end=backtest_settings.end)
    if dataset.empty:
        raise RuntimeError("Dataset is empty after BACKTEST_START/END filtering.")

    wf_pipeline = WalkForwardPipeline(train_cfg=train_cfg)
    wf_result = wf_pipeline.run(dataset)
    WalkForwardPipeline.save_artifacts(wf_result.predictions, wf_result.fold_details, train_cfg)
    lookup = {
        (_prediction_time(row), str(row.symbol)): (float(row.p_short), float(row.p_long))
        for row in wf_result.predictions.itertuples(index=False)
    }
    wf_model = WalkForwardPredictionModel(lookup=lookup, feature_columns=wf_result.feature_columns)

    frames = {symbol: _load_symbol_frame(backtest_settings, symbol) for symbol in symbols}
    strict_predictions = (
        backtest_settings.strict_oos
        or os.getenv("WF_REPLAY_STRICT_PREDICTIONS", "0").strip().lower() in ("1", "true", "yes")
    )
    start_bar_idx = max(0, int(backtest_settings.skip_initial_bars))
    if strict_predictions:
        first_indices: list[int] = []
        for symbol, frame in frames.items():
            symbol_predictions = wf_result.predictions.loc[wf_result.predictions["symbol"].astype(str) == symbol]
            if symbol_predictions.empty:
                raise RuntimeError(f"Walk-forward produced no OOS predictions for {symbol}.")
            prediction_time_col = "decision_time" if "decision_time" in symbol_predictions.columns else "timestamp"
            replay_time_col = "decision_time" if "decision_time" in frame.columns else "timestamp"
            first_pred_ts = pd.Timestamp(symbol_predictions[prediction_time_col].min())
            matches = frame.index[pd.to_datetime(frame[replay_time_col], errors="coerce") >= first_pred_ts].tolist()
            if not matches:
                raise RuntimeError(f"No replay rows at or after first OOS prediction timestamp {first_pred_ts} for {symbol}.")
            first_indices.append(int(matches[0]))
        if first_indices:
            start_bar_idx = max(start_bar_idx, max(first_indices))

    exchange = SimulatedExchange(
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    data_provider = HistoricalMultiSymbolReplayProvider(frames, skip_initial_bars=start_bar_idx)
    reporter = BacktestReplayReporter(
        config=backtest_settings,
        portfolio=portfolio,
        charts_dir=backtest_settings.charts_dir,
    )
    notifier = SilentNotifier()
    engine = build_trading_engine(
        settings=trading,
        exchange=exchange,
        data_provider=data_provider,
        model=wf_model,
        portfolio=portfolio,
        notifier=notifier,
    )
    print(
        f"TradingEngine Walk-forward OOS backtest | symbols={','.join(symbols)} | "
        f"rows={sum(len(frame) for frame in frames.values())} | "
        f"predictions={len(wf_result.predictions)} | folds={len(wf_result.fold_details)} | "
        f"first_bar_index={start_bar_idx} | strict_predictions={int(strict_predictions)} | "
        f"capital={trading.initial_capital}"
    )
    print("-" * 80)

    for ticks in data_provider.iter_replay_batches():
        await engine.process_market_batch(ticks)
        reporter.flush_trade_events()
        marks = {tick.symbol: float(tick.close if tick.close is not None else tick.price) for tick in ticks}
        reporter.record_equity(ticks[0].ts, marks)

    final_ts = data_provider.last_timestamp
    final_closes = data_provider.last_closes
    if final_ts is not None and final_closes:
        await close_all_positions_at_market(
            engine=engine,
            portfolio=portfolio,
            exchange=exchange,
            market_prices=final_closes,
            ts=final_ts,
        )
        reporter.flush_trade_events()
        reporter.record_equity(final_ts, final_closes)

    snap = portfolio.get_state_snapshot()
    print(
        f"Done | trades={snap['trades_count']} | closed={snap['closed_trades_count']} | "
        f"realized_pnl={snap['realized_pnl']:.4f}"
    )
    report = reporter.finish()
    export_closed_trades_csv(
        report,
        Path(backtest_settings.charts_dir) / "wf_trading_engine_closed_trades.csv",
        source="trading_engine",
    )


if __name__ == "__main__":
    asyncio.run(main())
