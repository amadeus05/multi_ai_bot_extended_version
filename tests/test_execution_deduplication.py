import asyncio

import pandas as pd

from application.execution_event_deduplicator import ExecutionEventDeduplicator
from application.trading_engine import TradingEngine
from core.types.events import FillEvent
from core.types.enums import OrderSide
from domain.portfolio.portfolio_manager import PortfolioManager


def fill_event(*, event_id: str | None = "fill-1") -> FillEvent:
    return FillEvent(
        ts=pd.Timestamp("2024-01-01T00:00:00"),
        order_id="order-1",
        client_order_id="client-1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=100.0,
        fee=0.0,
        event_id=event_id,
    )


def test_deduplicator_rejects_repeated_explicit_event_id() -> None:
    dedupe = ExecutionEventDeduplicator()
    event = fill_event(event_id="same-fill")

    assert dedupe.should_process(event) is True
    assert dedupe.should_process(event) is False


def test_deduplicator_fingerprints_fills_without_explicit_event_id() -> None:
    dedupe = ExecutionEventDeduplicator()
    event = fill_event(event_id=None)

    assert dedupe.should_process(event) is True
    assert dedupe.should_process(event) is False


class NoopRisk:
    pass


class NoopDependency:
    pass


def make_engine(portfolio: PortfolioManager) -> TradingEngine:
    return TradingEngine(
        exchange=NoopDependency(),
        data_provider=NoopDependency(),
        model=NoopDependency(),
        strategy=NoopDependency(),
        risk_manager=NoopRisk(),
        portfolio=portfolio,
        execution=NoopDependency(),
    )


def test_engine_does_not_apply_duplicate_fill_twice() -> None:
    portfolio = PortfolioManager(cash={"USDT": 1000.0})
    engine = make_engine(portfolio)
    event = fill_event(event_id="duplicate-fill")

    asyncio.run(engine.process_event(event))
    asyncio.run(engine.process_event(event))

    position = portfolio.get_position("BTC/USDT")
    assert position is not None
    assert position.amount == 1.0
    assert len(portfolio.trades) == 1
