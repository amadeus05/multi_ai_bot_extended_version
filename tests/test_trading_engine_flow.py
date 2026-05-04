import asyncio

import pandas as pd
import pytest

from application.trading_engine import ENGINE_META_EXIT_REASON, TradingEngine
from core.interfaces.notifier import Notifier
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Position, Tick
from core.types.enums import OrderSide, PositionSide
from core.types.events import ExitHeartbeatEvent, FillEvent, MarketEvent
from core.types.notifications import SignalNotification, SystemNotification, TradeExitNotification
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange


class FakeDataProvider:
    def __init__(self) -> None:
        self.warmup_calls: list[tuple[str, int]] = []

    async def warmup(self, symbol: str, bars: int) -> pd.DataFrame:
        self.warmup_calls.append((symbol, bars))
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01T00:00:00")],
                "symbol": [symbol],
                "close": [100.0],
            }
        )

    def subscribe(self, symbol, callback) -> None:
        pass

    async def run(self) -> None:
        pass


class FakeModel:
    def required_bars(self) -> int:
        return 5

    def predict(self, features: pd.DataFrame) -> dict:
        return {"score": 0.5, "p_long": 0.72, "p_short": 0.28}


class FakeStrategy:
    def on_prediction(self, tick: Tick, prediction: dict, portfolio: PortfolioManager):
        meta = {"score": prediction["score"]}
        if "barrier_stop_pct" in prediction:
            meta["barrier_stop_pct"] = prediction["barrier_stop_pct"]
        if "barrier_take_pct" in prediction:
            meta["barrier_take_pct"] = prediction["barrier_take_pct"]
        return Order(
            symbol=tick.symbol,
            side=OrderSide.BUY,
            amount=0.0,
            price=tick.price,
            meta=meta,
        )


class FakeRisk:
    def __init__(self) -> None:
        self.checked_orders: list[Order] = []
        self.registered_results: list[dict] = []

    def set_current_bar(self, _bar_ts) -> None:
        pass

    def check(self, order: Order, portfolio: PortfolioManager, exchange):
        self.checked_orders.append(order)
        order.amount = 2.0
        return order

    def register_trade_result(self, symbol: str, pnl: float, bar_ts, stop_loss_hit: bool = False) -> None:
        self.registered_results.append(
            {
                "symbol": symbol,
                "pnl": pnl,
                "bar_ts": bar_ts,
                "stop_loss_hit": stop_loss_hit,
            }
        )


class FakeBarrierPolicy:
    def barriers_for_last_row(self, _frame: pd.DataFrame):
        return 0.01, 0.02


class FakeExitManager:
    def __init__(self, price: float = 95.0, reason: str = "SL") -> None:
        self.price = price
        self.reason = reason
        self.calls: list[dict] = []

    def check_causal_exit(self, **kwargs):
        self.calls.append(kwargs)
        return self.price, self.reason


class NoopExchange:
    pass


class NoopExecution:
    pass


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.signals: list[SignalNotification] = []
        self.trade_exits: list[TradeExitNotification] = []
        self.systems: list[SystemNotification] = []
        self.errors: list[tuple[str, Exception, Order | None]] = []

    async def notify_signal(self, notification: SignalNotification) -> None:
        self.signals.append(notification)

    async def notify_trade_exit(self, notification: TradeExitNotification) -> None:
        self.trade_exits.append(notification)

    async def notify_system(self, notification: SystemNotification) -> None:
        self.systems.append(notification)

    async def notify_error(self, where: str, error: Exception, order: Order | None = None) -> None:
        self.errors.append((where, error, order))


def make_tick(symbol: str = "BTC/USDT") -> Tick:
    return Tick(
        symbol=symbol,
        ts=pd.Timestamp("2024-01-01T00:00:00"),
        bid=100.0,
        ask=100.0,
        price=100.0,
        volume=1.0,
        open=100.0,
        high=101.0,
        low=94.0,
    )


def make_engine(
    *,
    portfolio: PortfolioManager | None = None,
    data: FakeDataProvider | None = None,
    risk: FakeRisk | None = None,
    exit_manager=None,
    notifier: Notifier | None = None,
    exchange=None,
    execution=None,
) -> TradingEngine:
    return TradingEngine(
        exchange=exchange or NoopExchange(),
        data_provider=data or FakeDataProvider(),
        model=FakeModel(),
        strategy=FakeStrategy(),
        risk_manager=risk or FakeRisk(),
        portfolio=portfolio or PortfolioManager(cash={"USDT": 1000.0}),
        execution=execution or NoopExecution(),
        exit_manager=exit_manager,
        barrier_policy=FakeBarrierPolicy(),
        notifier=notifier,
    )


def test_market_event_without_position_builds_entry_command_with_prediction_metadata() -> None:
    data = FakeDataProvider()
    risk = FakeRisk()
    engine = make_engine(data=data, risk=risk)

    commands = asyncio.run(engine.process_event(MarketEvent(make_tick())))

    assert data.warmup_calls == [("BTC/USDT", 5)]
    assert len(commands) == 1
    command = commands[0]
    assert isinstance(command, PlaceOrderCommand)
    assert command.reason == "ENTRY"
    assert command.order.side == OrderSide.BUY
    assert command.order.amount == pytest.approx(2.0)
    assert command.order.meta["barrier_stop_pct"] == pytest.approx(0.01)
    assert command.order.meta["barrier_take_pct"] == pytest.approx(0.02)
    assert command.order.meta["p_long"] == pytest.approx(0.72)
    assert command.order.meta["p_short"] == pytest.approx(0.28)
    assert command.order.meta["signal_gap"] == pytest.approx(0.44)
    assert command.order.meta["direction_prob"] == pytest.approx(0.72)
    assert risk.checked_orders == [command.order]


def test_valid_entry_signal_is_notified_after_risk_check() -> None:
    notifier = RecordingNotifier()
    engine = make_engine(notifier=notifier)

    commands = asyncio.run(engine.process_event(MarketEvent(make_tick())))

    assert len(commands) == 1
    assert len(notifier.signals) == 1
    signal = notifier.signals[0]
    assert signal.signal_id == 1
    assert signal.symbol == "BTC/USDT"
    assert signal.side == "LONG"
    assert signal.entry_price == pytest.approx(100.0)
    assert signal.amount == pytest.approx(2.0)
    assert signal.stop_price == pytest.approx(99.0)
    assert signal.take_price == pytest.approx(102.0)
    assert signal.p_long == pytest.approx(0.72)
    assert signal.p_short == pytest.approx(0.28)
    assert commands[0].order.meta["signal_number"] == 1


def test_exit_check_runs_before_entry_and_returns_close_command_for_long_position() -> None:
    data = FakeDataProvider()
    exit_manager = FakeExitManager(price=94.0, reason="SL")
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.5,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    engine = make_engine(portfolio=portfolio, data=data, exit_manager=exit_manager)

    commands = asyncio.run(engine.process_event(MarketEvent(make_tick())))

    assert data.warmup_calls == []
    assert len(commands) == 1
    command = commands[0]
    assert command.reason == "SL"
    assert command.continue_with_entry is True
    assert command.source_tick == make_tick()
    assert command.order.side == OrderSide.SELL
    assert command.order.amount == pytest.approx(1.5)
    assert command.order.price == pytest.approx(94.0)
    assert command.order.meta[ORDER_META_FILL_PRICE_FINAL] is True
    assert command.order.meta[ENGINE_META_EXIT_REASON] == "SL"


def test_exit_command_for_short_position_uses_buy_side() -> None:
    exit_manager = FakeExitManager(price=106.0, reason="TP")
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.SHORT,
                amount=2.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    engine = make_engine(portfolio=portfolio, exit_manager=exit_manager)

    commands = asyncio.run(engine.process_event(MarketEvent(make_tick())))

    assert len(commands) == 1
    assert commands[0].order.side == OrderSide.BUY
    assert commands[0].order.amount == pytest.approx(2.0)


def test_exit_heartbeat_checks_only_exits_without_entry_warmup() -> None:
    data = FakeDataProvider()
    exit_manager = FakeExitManager(price=102.0, reason="TP")
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    engine = make_engine(portfolio=portfolio, data=data, exit_manager=exit_manager)

    commands = asyncio.run(engine.process_event(ExitHeartbeatEvent(make_tick())))

    assert data.warmup_calls == []
    assert len(commands) == 1
    command = commands[0]
    assert command.reason == "TP"
    assert command.continue_with_entry is False
    assert command.source_tick is None
    assert command.order.side == OrderSide.SELL
    assert command.order.price == pytest.approx(102.0)


def test_exit_heartbeat_executes_paper_exit_and_closes_position() -> None:
    notifier = RecordingNotifier()
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    tick = Tick(
        symbol="BTC/USDT",
        ts=pd.Timestamp("2024-01-01T00:01:00"),
        bid=102.5,
        ask=102.5,
        price=102.5,
        volume=1.0,
        open=100.0,
        high=102.5,
        low=99.5,
        close=102.5,
    )
    engine = make_engine(
        portfolio=portfolio,
        exchange=SimulatedExchange(commission=0.0, slippage=0.0),
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=0.0),
        notifier=notifier,
    )

    commands = asyncio.run(engine.process_event(ExitHeartbeatEvent(tick)))
    events = asyncio.run(engine.execute_commands(commands))

    assert portfolio.get_position("BTC/USDT") is None
    assert len(events) == 2
    assert len(notifier.trade_exits) == 1
    assert notifier.trade_exits[0].reason == "TP"
    assert notifier.trade_exits[0].exit_price == pytest.approx(102.0)


def test_exit_fill_with_continue_entry_closes_position_then_builds_new_entry_command() -> None:
    risk = FakeRisk()
    tick = make_tick()
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    engine = make_engine(portfolio=portfolio, risk=risk)
    fill = FillEvent(
        ts=tick.ts,
        order_id="exit-1",
        client_order_id="client-exit-1",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=95.0,
        fee=0.0,
        meta={ENGINE_META_EXIT_REASON: "SL"},
        command_reason="SL",
        source_tick=tick,
        continue_with_entry=True,
        event_id="exit-fill-1",
    )

    commands = asyncio.run(engine.process_event(fill))

    assert portfolio.get_position("BTC/USDT") is None
    assert risk.registered_results[-1]["symbol"] == "BTC/USDT"
    assert risk.registered_results[-1]["stop_loss_hit"] is True
    assert len(commands) == 1
    assert commands[0].reason == "ENTRY"
    assert commands[0].order.side == OrderSide.BUY


def test_exit_fill_notifies_closed_trade_with_pnl_and_balance() -> None:
    notifier = RecordingNotifier()
    portfolio = PortfolioManager(
        cash={"USDT": 1000.0},
        positions=[
            Position(
                "BTC/USDT",
                PositionSide.LONG,
                amount=1.0,
                entry_price=100.0,
                meta={"barrier_stop_pct": 0.01, "barrier_take_pct": 0.02},
            )
        ],
    )
    engine = make_engine(portfolio=portfolio, notifier=notifier)
    fill = FillEvent(
        ts=pd.Timestamp("2024-01-01T01:00:00"),
        order_id="exit-1",
        client_order_id="client-exit-1",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=95.0,
        fee=0.0,
        meta={ENGINE_META_EXIT_REASON: "SL"},
        command_reason="SL",
        event_id="exit-fill-1",
    )

    commands = asyncio.run(engine.process_event(fill))

    assert commands == []
    assert len(notifier.trade_exits) == 1
    notification = notifier.trade_exits[0]
    assert notification.trade_number == 0
    assert notification.symbol == "BTC/USDT"
    assert notification.side == "LONG"
    assert notification.reason == "SL"
    assert notification.entry_price == pytest.approx(100.0)
    assert notification.exit_price == pytest.approx(95.0)
    assert notification.pnl_abs == pytest.approx(-5.0)
    assert notification.pnl_pct == pytest.approx(-0.05)
    assert notification.balance == pytest.approx(995.0)
