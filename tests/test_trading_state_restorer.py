import asyncio

import pandas as pd

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_state_restorer import ExchangeSnapshotStateRestorer, RestoreResult
from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.portfolio.portfolio_manager import PortfolioManager


class FakeExchange:
    async def get_balance(self, asset: str) -> float:
        assert asset == "USDT"
        return 1234.5

    async def get_positions(self) -> list[Position]:
        return [Position("BTC/USDT", PositionSide.LONG, amount=0.2, entry_price=30000.0)]


def test_exchange_snapshot_restorer_updates_in_memory_portfolio() -> None:
    portfolio = PortfolioManager(cash={"USDT": 100.0})
    restorer = ExchangeSnapshotStateRestorer(exchange=FakeExchange(), portfolio=portfolio)

    result = asyncio.run(restorer.restore_trading_state())

    assert result.restored is True
    assert result.source == "exchange_snapshot"
    assert result.positions_count == 1
    assert portfolio.cash["USDT"] == 1234.5
    assert portfolio.positions[0].symbol == "BTC/USDT"
    assert portfolio.positions[0].amount == 0.2


class FakeRestorer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def restore_trading_state(self) -> RestoreResult:
        self.calls.append("restore")
        return RestoreResult(restored=True, source="fake")


class FakeProvider:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        assert self.calls == ["restore"]
        self.calls.append("warmup")
        return pd.DataFrame([{"timestamp": pd.Timestamp("2024-01-01")}])

    def subscribe(self, symbol: str, callback) -> None:
        assert self.calls == ["restore", "warmup"]
        self.calls.append("subscribe")

    async def run(self) -> None:
        assert self.calls == ["restore", "warmup", "subscribe"]
        self.calls.append("run")


class FakeModel:
    def required_bars(self) -> int:
        return 1


class FakeEngine:
    def set_event_journal(self, journal) -> None:
        return None


def test_realtime_orchestrator_restores_before_warmup_and_subscribe() -> None:
    calls: list[str] = []
    orchestrator = RealtimeOrchestrator(
        engine=FakeEngine(),
        data_provider=FakeProvider(calls),
        model=FakeModel(),
        symbols=["BTC/USDT"],
        state_restorer=FakeRestorer(calls),
    )

    asyncio.run(orchestrator.run())

    assert calls == ["restore", "warmup", "subscribe", "run"]
