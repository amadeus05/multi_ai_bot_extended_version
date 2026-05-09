import asyncio

import pandas as pd

from application.realtime_orchestrator import RealtimeOrchestrator
from core.types.domain_types import Tick


class FakeEngine:
    pass


class FakeModel:
    def required_bars(self) -> int:
        return 1


class FakeProvider:
    async def warmup(self, symbol: str, bars: int):
        return pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "symbol": [symbol]})

    def subscribe(self, symbol: str, callback) -> None:
        pass

    async def run(self) -> None:
        pass


class FakeRuntime:
    def __init__(self) -> None:
        self.batches = []
        self.events = []

    async def publish_market_batch(self, ticks):
        self.batches.append(list(ticks))

    async def publish(self, event) -> None:
        self.events.append(event)


def test_realtime_orchestrator_batches_symbols_by_timestamp() -> None:
    runtime = FakeRuntime()
    orchestrator = RealtimeOrchestrator(
        engine=FakeEngine(),
        data_provider=FakeProvider(),
        model=FakeModel(),
        symbols=["BTC/USDT", "ETH/USDT"],
        runtime=runtime,
    )

    asyncio.run(orchestrator._publish_tick(_tick("ETH/USDT", "2024-01-01T01:00:00")))
    assert runtime.batches == []

    asyncio.run(orchestrator._publish_tick(_tick("BTC/USDT", "2024-01-01T01:00:00")))

    assert len(runtime.batches) == 1
    assert [tick.symbol for tick in runtime.batches[0]] == ["BTC/USDT", "ETH/USDT"]


def test_realtime_orchestrator_flushes_partial_batch_on_next_timestamp() -> None:
    runtime = FakeRuntime()
    orchestrator = RealtimeOrchestrator(
        engine=FakeEngine(),
        data_provider=FakeProvider(),
        model=FakeModel(),
        symbols=["BTC/USDT", "ETH/USDT"],
        runtime=runtime,
    )

    asyncio.run(orchestrator._publish_tick(_tick("ETH/USDT", "2024-01-01T01:00:00")))
    asyncio.run(orchestrator._publish_tick(_tick("ETH/USDT", "2024-01-01T02:00:00")))

    assert len(runtime.batches) == 1
    assert [tick.symbol for tick in runtime.batches[0]] == ["ETH/USDT"]


def _tick(symbol: str, ts: str) -> Tick:
    return Tick(
        symbol=symbol,
        ts=pd.Timestamp(ts),
        bid=100.0,
        ask=100.0,
        price=100.0,
        volume=1.0,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
    )
