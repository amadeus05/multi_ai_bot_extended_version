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
from application.trading_runtime_loop import TradingRuntimeLoop
from application.trading_runtime_factory import build_trading_engine
from core.config.loader import load_backtest_settings
from core.config.train_config import TrainConfig
from core.types.events import MarketEvent
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.data_providers.historical_replay_provider import HistoricalReplayProvider
from infrastructure.notifications import SilentNotifier
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
    if len(symbols) != 1:
        raise ValueError(
            "TradingEngine walk-forward backtest currently supports exactly one symbol. "
            "Set SYMBOLS to one instrument, for example SYMBOLS=BTC/USDT."
        )
    if backtest_settings.strict_oos and not backtest_settings.start:
        raise ValueError("BACKTEST_STRICT_OOS=1 requires BACKTEST_START.")

    symbol = symbols[0]
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
        (pd.Timestamp(row.timestamp), str(row.symbol)): (float(row.p_short), float(row.p_long))
        for row in wf_result.predictions.itertuples(index=False)
    }
    wf_model = WalkForwardPredictionModel(lookup=lookup, feature_columns=wf_result.feature_columns)

    frame = _load_symbol_frame(backtest_settings, symbol)
    strict_predictions = (
        backtest_settings.strict_oos
        or os.getenv("WF_REPLAY_STRICT_PREDICTIONS", "0").strip().lower() in ("1", "true", "yes")
    )
    start_bar_idx = max(0, int(backtest_settings.skip_initial_bars))
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
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    data_provider = HistoricalReplayProvider(symbol, frame, skip_initial_bars=start_bar_idx)
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
    runtime = TradingRuntimeLoop(engine, journal=None, notifier=notifier)

    print(
        f"TradingEngine Walk-forward OOS backtest | {symbol} | rows={len(frame)} | "
        f"predictions={len(wf_result.predictions)} | folds={len(wf_result.fold_details)} | "
        f"first_bar_index={start_bar_idx} | strict_predictions={int(strict_predictions)} | "
        f"capital={trading.initial_capital}"
    )
    print("-" * 80)

    for tick in data_provider.iter_replay_ticks():
        await runtime.process_once(MarketEvent(tick))
        reporter.flush_trade_events()
        mark = float(tick.close if tick.close is not None else tick.price)
        reporter.record_equity(tick.ts, {symbol: mark})

    final_ts = data_provider.last_timestamp
    final_close = data_provider.last_close
    if final_ts is not None and final_close is not None:
        await close_all_positions_at_market(
            engine=engine,
            portfolio=portfolio,
            exchange=exchange,
            market_prices={symbol: float(final_close)},
            ts=final_ts,
        )
        reporter.flush_trade_events()
        reporter.record_equity(final_ts, {symbol: float(final_close)})

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
