from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.trading_engine import TradingEngine
from application.trading_runtime_loop import TradingRuntimeLoop
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model
from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide
from core.types.events import FillEvent, MarketEvent, OrderAcceptedEvent, OrderCancelledEvent
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


class StaticDataProvider(DataProvider):
    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "open": 100.0,
                    "high": 103.0,
                    "low": 100.0,
                    "close": 102.0,
                    "volume": 1.0,
                    "symbol": symbol,
                }
            ]
        )

    def subscribe(self, symbol: str, callback) -> None:
        self.callback = callback

    async def run(self) -> None:
        return None


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
            meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.01},
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
    await smoke_simulated_exchange_lifecycle()
    await smoke_entry_and_intrabar_exit()
    await smoke_exit_then_reenter_same_tick()
    print("event runtime smoke ok")


async def smoke_simulated_exchange_lifecycle() -> None:
    exchange = SimulatedExchange(commission=0.0, slippage=0.0, leverage=1.0)
    execution = ExecutionService()
    order = Order(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0, price=100.0)
    events = await execution.execute(
        PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-02 00:00:00")),
        exchange,
    )

    assert [type(event) for event in events] == [OrderAcceptedEvent, FillEvent], events
    assert events[0].order_id == events[1].order_id
    assert order.id == events[0].order_id

    cancel_events = await execution.execute(
        CancelOrderCommand(order_id=events[0].order_id, reason="TEST", ts=pd.Timestamp("2024-01-02 00:01:00")),
        exchange,
    )
    assert [type(event) for event in cancel_events] == [OrderCancelledEvent], cancel_events
    assert cancel_events[0].order_id == events[0].order_id


async def smoke_entry_and_intrabar_exit() -> None:
    symbol = "BTC/USDT"
    ts = pd.Timestamp("2024-01-02 00:00:00")
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    engine = TradingEngine(
        exchange=SimulatedExchange(commission=0.0, slippage=0.0, leverage=1.0),
        data_provider=StaticDataProvider(),
        model=StaticModel(),
        strategy=LongOnlyStrategy(),
        risk_manager=PassThroughRisk(),
        portfolio=portfolio,
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=0.0),
        barrier_policy=NoBarrierPolicy(),
    )
    runtime = TradingRuntimeLoop(engine)

    tick = Tick(
        symbol=symbol,
        ts=ts,
        bid=100.0,
        ask=100.0,
        price=100.0,
        volume=1.0,
        open=100.0,
        high=103.0,
        low=100.0,
        close=102.0,
    )
    execution_events = await runtime.process_once(MarketEvent(tick))

    assert len(execution_events) == 4, execution_events
    assert len(portfolio.trades) == 2, portfolio.get_state_snapshot()
    assert len(portfolio.get_open_positions()) == 0, portfolio.get_state_snapshot()
    assert len(portfolio.closed_trade_results) == 1, portfolio.closed_trade_results
    assert portfolio.closed_trade_results[0]["reason"] == "TP", portfolio.closed_trade_results
    assert portfolio.trade_events[-1]["reason_suffix"] == "INTRA-BAR", portfolio.trade_events
    assert portfolio.trades[0].ts == ts and portfolio.trades[1].ts == ts


async def smoke_exit_then_reenter_same_tick() -> None:
    symbol = "BTC/USDT"
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    engine = TradingEngine(
        exchange=SimulatedExchange(commission=0.0, slippage=0.0, leverage=1.0),
        data_provider=StaticDataProvider(),
        model=StaticModel(),
        strategy=LongOnlyStrategy(),
        risk_manager=PassThroughRisk(),
        portfolio=portfolio,
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=0.0),
        barrier_policy=NoBarrierPolicy(),
    )
    runtime = TradingRuntimeLoop(engine)

    first_tick = Tick(
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
    )
    await runtime.process_once(MarketEvent(first_tick))
    assert len(portfolio.get_open_positions()) == 1, portfolio.get_state_snapshot()

    second_tick = Tick(
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
    )
    await runtime.process_once(MarketEvent(second_tick))

    assert len(portfolio.closed_trade_results) == 2, portfolio.closed_trade_results
    assert portfolio.closed_trade_results[0]["reason"] == "TP", portfolio.closed_trade_results
    assert len(portfolio.get_open_positions()) == 0, portfolio.get_state_snapshot()
    assert [event["type"] for event in portfolio.trade_events] == ["OPEN", "CLOSE", "OPEN", "CLOSE"]


if __name__ == "__main__":
    asyncio.run(main())
