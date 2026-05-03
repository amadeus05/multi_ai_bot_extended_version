from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.event_journal import EventJournal
from application.trading_engine import TradingEngine
from application.trading_runtime_loop import TradingRuntimeLoop
from core.config.storage_config import StorageSettings
from core.interfaces.data_provider import DataProvider
from core.interfaces.model import Model
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide
from core.types.events import MarketEvent
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from infrastructure.storage.storage_factory import build_event_repository


class StaticDataProvider(DataProvider):
    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
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


class NoBarrierPolicy:
    def barriers_for_last_row(self, frame: pd.DataFrame):
        return None


async def main() -> None:
    db_path = Path("data") / "smoke_event_journal.sqlite"
    if db_path.exists():
        db_path.unlink()

    repo = build_event_repository(
        StorageSettings(driver="sqlite", sqlite_path=str(db_path), events_table="trading_events")
    )
    journal = EventJournal(repo, session_id="journal-smoke", source="smoke")
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    engine = TradingEngine(
        exchange=SimulatedExchange(commission=0.0, slippage=0.0, leverage=1.0),
        data_provider=StaticDataProvider(),
        model=StaticModel(),
        strategy=LongOnlyStrategy(),
        risk_manager=PassThroughRisk(),
        portfolio=portfolio,
        execution=ExecutionService(),
        barrier_policy=NoBarrierPolicy(),
    )
    runtime = TradingRuntimeLoop(engine, journal=journal)

    tick = Tick(
        symbol="BTC/USDT",
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
    await runtime.process_once(MarketEvent(tick))

    rows = [record async for record in repo.iter_session("journal-smoke")]
    assert [row.kind for row in rows] == [
        "MarketEvent",
        "PlaceOrderCommand",
        "OrderAcceptedEvent",
        "FillEvent",
    ], rows
    assert rows[0].source == "runtime", rows[0]
    assert rows[1].source == "engine", rows[1]
    assert rows[2].source == "execution", rows[2]
    assert rows[3].payload["symbol"] == "BTC/USDT", rows[3]

    print("event journal smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
