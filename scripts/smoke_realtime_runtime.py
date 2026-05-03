from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_engine import TradingEngine
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


class FiniteRealtimeProvider(DataProvider):
    def __init__(self, ticks: list[Tick]) -> None:
        self._ticks = ticks
        self._subs = {}
        self.bootstrap_calls = 0

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        self.bootstrap_calls += 1
        return pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.0,
                    "volume": 1.0,
                    "symbol": symbol,
                }
            ]
        )

    def subscribe(self, symbol: str, callback) -> None:
        self._subs[symbol] = callback

    async def run(self) -> None:
        for tick in self._ticks:
            callback = self._subs[tick.symbol]
            result = callback(tick)
            if asyncio.iscoroutine(result):
                await result


class StaticModel(Model):
    def predict(self, features: pd.DataFrame) -> dict:
        return {"p_long": 0.8, "p_short": 0.2, "score": 0.8}

    def required_bars(self) -> int:
        return 1


class LongOnlyStrategy:
    def on_prediction(self, tick: Tick, prediction: dict, portfolio: PortfolioManager) -> Order:
        return Order(
            symbol=tick.symbol,
            side=OrderSide.BUY,
            amount=1.0,
            price=tick.price,
            meta={"barrier_stop_pct": 0.02, "barrier_take_pct": 0.02},
        )


class PassThroughRisk:
    def set_current_bar(self, bar_ts) -> None:
        self.current_bar_ts = bar_ts

    def check(self, order: Order, portfolio: PortfolioManager, exchange) -> Order:
        return order

    def register_trade_result(self, symbol: str, pnl: float, bar_ts, stop_loss_hit: bool = False) -> None:
        self.last_result = (symbol, pnl, bar_ts, stop_loss_hit)


class NoBarrierPolicy:
    def barriers_for_last_row(self, frame: pd.DataFrame):
        return None


async def main() -> None:
    symbol = "BTC/USDT"
    ticks = [
        Tick(
            symbol=symbol,
            ts=pd.Timestamp("2024-01-02 00:00:00"),
            bid=100.0,
            ask=100.0,
            price=100.0,
            volume=1.0,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
        ),
        Tick(
            symbol=symbol,
            ts=pd.Timestamp("2024-01-02 01:00:00"),
            bid=100.0,
            ask=100.0,
            price=100.0,
            volume=1.0,
            open=100.0,
            high=103.0,
            low=99.5,
            close=102.0,
        ),
    ]
    provider = FiniteRealtimeProvider(ticks)
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    engine = TradingEngine(
        exchange=SimulatedExchange(commission=0.0, slippage=0.0, leverage=1.0),
        data_provider=provider,
        model=StaticModel(),
        strategy=LongOnlyStrategy(),
        risk_manager=PassThroughRisk(),
        portfolio=portfolio,
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=0.0),
        barrier_policy=NoBarrierPolicy(),
    )
    orchestrator = RealtimeOrchestrator(
        engine=engine,
        data_provider=provider,
        model=StaticModel(),
        symbols=[symbol],
        warmup_bars=1,
    )

    await orchestrator.run()

    assert provider.bootstrap_calls >= 1
    assert len(portfolio.closed_trade_results) == 2, portfolio.closed_trade_results
    assert [row["reason"] for row in portfolio.closed_trade_results] == ["TP", "TP"], portfolio.closed_trade_results
    assert len(portfolio.get_open_positions()) == 0, portfolio.get_state_snapshot()
    assert [event["type"] for event in portfolio.trade_events] == ["OPEN", "CLOSE", "OPEN", "CLOSE"]

    print("realtime runtime smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
