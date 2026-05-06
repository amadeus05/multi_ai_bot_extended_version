from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.backtest_position_closer import close_all_positions_at_market
from application.backtest_replay_reporter import BacktestReplayReporter
from application.backtest_trade_export import export_closed_trades_csv
from application.inference.walk_forward_model import WalkForwardPredictionModel
from application.trading_runtime_factory import build_trading_engine
from application.trading_runtime_loop import TradingRuntimeLoop
from core.config.loader import load_backtest_settings
from core.types.domain_types import Tick
from core.types.events import MarketEvent
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from infrastructure.notifications import SilentNotifier


def _filter_backtest_frame(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if start:
        out = out.loc[out["timestamp"] >= pd.to_datetime(start, errors="coerce")]
    if end:
        out = out.loc[out["timestamp"] <= pd.to_datetime(end, errors="coerce")]
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
        raise FileNotFoundError(f"Dataset for {symbol} not found: {dataset_path}. Run dataset pipeline first.")
    frame = pd.read_parquet(dataset_path)
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    else:
        frame["symbol"] = frame["symbol"].fillna(symbol).astype(str)
    frame = _filter_backtest_frame(frame, start=settings.start, end=settings.end)
    if frame.empty:
        raise ValueError(f"No rows remain for {symbol} after BACKTEST_START/END filtering.")
    return frame


class MultiSymbolReplayProvider:
    def __init__(self, frames: dict[str, pd.DataFrame], *, cursor_by_symbol: dict[str, int] | None = None) -> None:
        self._frames = frames
        self._cursor_by_symbol = cursor_by_symbol or {symbol: -1 for symbol in frames}

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        frame = self._frames.get(symbol, pd.DataFrame())
        if frame.empty:
            return pd.DataFrame()
        cursor_idx = self._cursor_by_symbol.get(symbol, -1)
        bars = max(1, int(bars))
        if cursor_idx < 0:
            return frame.iloc[: min(bars, len(frame))].copy()
        end = min(cursor_idx, len(frame) - 1)
        start = max(0, end - bars + 1)
        return frame.iloc[start : end + 1].copy()

    def subscribe(self, symbol, callback):
        return None

    async def run(self) -> None:
        return None

    def iter_replay_ticks(self, *, start_index_by_symbol: dict[str, int]) -> list[Tick]:
        ticks: list[tuple[pd.Timestamp, str, int, Tick]] = []
        for symbol, frame in self._frames.items():
            if frame.empty:
                continue
            start_idx = max(0, int(start_index_by_symbol.get(symbol, 0)))
            if start_idx >= len(frame):
                continue
            for idx in range(start_idx, len(frame) - 1):
                next_row = frame.iloc[idx + 1]
                ts = pd.Timestamp(next_row["timestamp"])
                if hasattr(ts, "tz_convert") and ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
                px = float(next_row.get("open", next_row.get("close", 0.0)))
                tick = Tick(
                    symbol=symbol,
                    ts=ts,
                    bid=px,
                    ask=px,
                    price=px,
                    volume=float(next_row.get("volume", 0.0)),
                    open=float(next_row.get("open", px)),
                    high=float(next_row.get("high", px)),
                    low=float(next_row.get("low", px)),
                    close=float(next_row.get("close", px)),
                )
                ticks.append((ts, symbol, idx, tick))
        ticks.sort(key=lambda item: (item[0], item[1]))
        for _ts, symbol, idx, tick in ticks:
            self._cursor_by_symbol[symbol] = idx
            yield tick

    @property
    def last_prices(self) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol, frame in self._frames.items():
            if not frame.empty:
                prices[symbol] = float(frame["close"].iloc[-1])
        return prices

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        timestamps = [pd.Timestamp(frame["timestamp"].iloc[-1]) for frame in self._frames.values() if not frame.empty]
        return max(timestamps) if timestamps else None


async def main() -> None:
    settings = load_backtest_settings()
    trading = settings.trading
    symbols = list(trading.symbols)
    model_name = "candle_lstm_target"
    predictions_path = Path(f"models/{model_name}_walk_forward_oos_predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Candle LSTM OOS predictions not found: {predictions_path}. Run `python runners/run_train_candle_lstm.py` first."
        )
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    predictions = _filter_backtest_frame(predictions, start=settings.start, end=settings.end)
    predictions = predictions.loc[predictions["symbol"].astype(str).isin(symbols)].copy()
    if predictions.empty:
        raise RuntimeError(f"No Candle LSTM OOS predictions for configured SYMBOLS={symbols!r}.")
    lookup = {
        (pd.Timestamp(row.timestamp), str(row.symbol)): (float(row.p_short), float(row.p_long))
        for row in predictions.itertuples(index=False)
    }
    wf_model = WalkForwardPredictionModel(lookup=lookup, feature_columns=[])

    frames = {symbol: _load_symbol_frame(settings, symbol) for symbol in symbols}
    start_index_by_symbol: dict[str, int] = {}
    for symbol in symbols:
        symbol_predictions = predictions.loc[predictions["symbol"].astype(str) == symbol]
        if symbol_predictions.empty:
            continue
        first_pred_ts = pd.Timestamp(symbol_predictions["timestamp"].min())
        frame = frames[symbol]
        first_pred_matches = frame.index[pd.to_datetime(frame["timestamp"], errors="coerce") >= first_pred_ts].tolist()
        start_idx = max(0, int(settings.skip_initial_bars))
        if first_pred_matches:
            start_idx = max(start_idx, int(first_pred_matches[0]))
        start_index_by_symbol[symbol] = start_idx

    exchange = SimulatedExchange(
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    data_provider = MultiSymbolReplayProvider(frames)
    reporter = BacktestReplayReporter(config=settings, portfolio=portfolio, charts_dir=settings.charts_dir)
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
        f"Candle LSTM Walk-forward OOS backtest | symbols={','.join(symbols)} | "
        f"predictions={len(predictions)} | replay_symbols={len(start_index_by_symbol)}"
    )
    for tick in data_provider.iter_replay_ticks(start_index_by_symbol=start_index_by_symbol):
        await runtime.process_once(MarketEvent(tick))
        reporter.flush_trade_events()
        mark = float(tick.close if tick.close is not None else tick.price)
        marks = dict(data_provider.last_prices)
        marks[tick.symbol] = mark
        reporter.record_equity(tick.ts, marks)
    final_ts = data_provider.last_timestamp
    final_prices = data_provider.last_prices
    if final_ts is not None and final_prices:
        await close_all_positions_at_market(
            engine=engine,
            portfolio=portfolio,
            exchange=exchange,
            market_prices=final_prices,
            ts=final_ts,
        )
        reporter.flush_trade_events()
        reporter.record_equity(final_ts, final_prices)
    report = reporter.finish()
    export_closed_trades_csv(
        report,
        Path(settings.charts_dir) / "candle_lstm_wf_closed_trades.csv",
        source="candle_lstm_walk_forward",
    )


if __name__ == "__main__":
    asyncio.run(main())
