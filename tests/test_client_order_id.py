import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Tick
from core.types.enums import OrderSide, OrderType
from domain.execution.client_order_id import MAX_CLIENT_ORDER_ID_LEN, deterministic_client_order_id, ensure_client_order_id


def make_tick(ts: str = "2024-01-01T00:00:00") -> Tick:
    return Tick(
        symbol="BTC/USDT",
        ts=pd.Timestamp(ts),
        bid=100.0,
        ask=100.0,
        price=100.0,
        volume=1.0,
    )


def make_command(*, side: OrderSide = OrderSide.BUY, ts: str = "2024-01-01T00:00:00") -> PlaceOrderCommand:
    order = Order(
        "BTC/USDT",
        side,
        amount=1.25,
        price=100.0,
        type=OrderType.LIMIT,
        meta={
            "signal_number": 7,
            "p_long": 0.7,
            "p_short": 0.3,
            "barrier_stop_pct": 0.01,
            "volatile_runtime_noise": "ignored",
        },
    )
    return PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp(ts), source_tick=make_tick(ts))


def test_deterministic_client_order_id_is_stable_for_same_intent() -> None:
    first = deterministic_client_order_id(make_command())
    second = deterministic_client_order_id(make_command())

    assert first == second
    assert first is not None
    assert len(first) <= MAX_CLIENT_ORDER_ID_LEN


def test_deterministic_client_order_id_ignores_runtime_signal_number() -> None:
    first = make_command()
    second = make_command()
    first.order.meta["signal_number"] = 1
    second.order.meta["signal_number"] = 999

    assert deterministic_client_order_id(first) == deterministic_client_order_id(second)


def test_deterministic_client_order_id_changes_for_different_intent() -> None:
    first = deterministic_client_order_id(make_command(side=OrderSide.BUY))
    second = deterministic_client_order_id(make_command(side=OrderSide.SELL))
    third = deterministic_client_order_id(make_command(ts="2024-01-01T01:00:00"))

    assert first != second
    assert first != third


def test_deterministic_client_order_id_uses_command_ts_when_source_tick_missing() -> None:
    command = make_command()
    command = PlaceOrderCommand(command.order, reason=command.reason, ts=pd.Timestamp("2024-01-01T00:00:00"))

    assert deterministic_client_order_id(command) is not None


def test_deterministic_client_order_id_returns_none_without_any_timestamp() -> None:
    command = make_command()
    command = PlaceOrderCommand(command.order, reason=command.reason)

    assert deterministic_client_order_id(command) is None


def test_ensure_client_order_id_preserves_existing_id() -> None:
    command = make_command()
    command.order.client_order_id = "already-set"

    ensure_client_order_id(command.order, command)

    assert command.order.client_order_id == "already-set"


def test_ensure_client_order_id_uses_uuid_fallback_when_intent_is_not_deterministic() -> None:
    command = make_command()
    command = PlaceOrderCommand(command.order, reason=command.reason)

    ensure_client_order_id(command.order, command)

    assert command.order.client_order_id is not None
    assert len(command.order.client_order_id) == 36
