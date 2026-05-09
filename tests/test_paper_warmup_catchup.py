import asyncio

import pandas as pd

from application.paper_warmup_catchup import PaperWarmupCatchup
from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.portfolio.portfolio_manager import PortfolioManager


class FakeEngine:
    def __init__(self) -> None:
        self.commands = []

    async def execute_commands(self, commands):
        self.commands.extend(commands)
        for command in commands:
            command.order.client_order_id = "client-exit"
        return [type("Accepted", (), {"event_id": "accepted"})(), type("Fill", (), {"event_id": "fill"})()]


class FakeExitManager:
    def __init__(self, close_on_low: float) -> None:
        self.close_on_low = close_on_low
        self.calls = []

    def check_causal_exit(self, **kwargs):
        self.calls.append(kwargs)
        if float(kwargs["next_low"]) <= self.close_on_low:
            return 1960.0, "TP"
        return None, None


def test_paper_warmup_catchup_replays_after_entry_and_stops_on_exit() -> None:
    portfolio = PortfolioManager(
        cash={"USDT": 100.0},
        positions=[
            Position(
                "ETH/USDT",
                PositionSide.SHORT,
                amount=0.25,
                entry_price=2000.0,
                meta={
                    "entry_ts": "2024-01-01T00:00:00",
                    "barrier_stop_pct": 0.01,
                    "barrier_take_pct": 0.02,
                },
            )
        ],
    )
    frame = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2024-01-01T00:00:00"), "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "volume": 1.0},
            {"timestamp": pd.Timestamp("2024-01-01T01:00:00"), "open": 2005.0, "high": 2008.0, "low": 1995.0, "close": 1998.0, "volume": 1.0},
            {"timestamp": pd.Timestamp("2024-01-01T02:00:00"), "open": 1998.0, "high": 2001.0, "low": 1950.0, "close": 1960.0, "volume": 1.0},
            {"timestamp": pd.Timestamp("2024-01-01T03:00:00"), "open": 1960.0, "high": 1970.0, "low": 1940.0, "close": 1950.0, "volume": 1.0},
        ]
    )
    engine = FakeEngine()
    exit_manager = FakeExitManager(close_on_low=1950.0)

    asyncio.run(
        PaperWarmupCatchup(
            portfolio=portfolio,
            engine=engine,
            exit_manager=exit_manager,
        ).run({"ETH/USDT": frame})
    )

    assert len(exit_manager.calls) == 2
    assert engine.commands[0].reason == "TP"
    assert engine.commands[0].order.client_order_id == "client-exit"


def test_paper_warmup_catchup_skips_when_entry_is_after_warmup() -> None:
    portfolio = PortfolioManager(
        cash={"USDT": 100.0},
        positions=[
            Position(
                "ETH/USDT",
                PositionSide.SHORT,
                amount=0.25,
                entry_price=2000.0,
                meta={
                    "entry_ts": "2024-01-02T00:00:00",
                    "barrier_stop_pct": 0.01,
                    "barrier_take_pct": 0.02,
                },
            )
        ],
    )
    frame = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2024-01-01T00:00:00"), "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "volume": 1.0},
        ]
    )
    engine = FakeEngine()
    exit_manager = FakeExitManager(close_on_low=1950.0)

    asyncio.run(
        PaperWarmupCatchup(
            portfolio=portfolio,
            engine=engine,
            exit_manager=exit_manager,
        ).run({"ETH/USDT": frame})
    )

    assert exit_manager.calls == []
    assert engine.commands == []


def test_paper_warmup_catchup_compares_timezone_aware_entry_with_naive_frame() -> None:
    portfolio = PortfolioManager(
        cash={"USDT": 100.0},
        positions=[
            Position(
                "ETH/USDT",
                PositionSide.SHORT,
                amount=0.25,
                entry_price=2000.0,
                meta={
                    "entry_ts": "2024-01-01T00:00:00+00:00",
                    "barrier_stop_pct": 0.01,
                    "barrier_take_pct": 0.02,
                },
            )
        ],
    )
    frame = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2024-01-01T00:00:00"), "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "volume": 1.0},
            {"timestamp": pd.Timestamp("2024-01-01T01:00:00"), "open": 2005.0, "high": 2008.0, "low": 1950.0, "close": 1960.0, "volume": 1.0},
        ]
    )
    engine = FakeEngine()
    exit_manager = FakeExitManager(close_on_low=1950.0)

    asyncio.run(
        PaperWarmupCatchup(
            portfolio=portfolio,
            engine=engine,
            exit_manager=exit_manager,
        ).run({"ETH/USDT": frame})
    )

    assert len(exit_manager.calls) == 1
    assert engine.commands[0].reason == "TP"
