import pytest

from core.types.domain_types import Order
from core.types.enums import OrderSide
from infrastructure.exchanges.bybit.bybit_execution_safety import (
    BybitExecutionSafety,
    BybitExecutionSafetyError,
)


def test_allows_order_when_no_safety_limits_are_set() -> None:
    safety = BybitExecutionSafety()
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0)

    safety.validate(order)


def test_kill_switch_rejects_every_order() -> None:
    safety = BybitExecutionSafety(kill_switch=True)
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0)

    with pytest.raises(BybitExecutionSafetyError, match="kill switch"):
        safety.validate(order)


def test_allowlist_rejects_unknown_symbol() -> None:
    safety = BybitExecutionSafety.from_values(allowlist_symbols=["BTC/USDT"])
    order = Order("ETH/USDT", OrderSide.BUY, amount=1.0, price=100.0)

    with pytest.raises(BybitExecutionSafetyError, match="allowlist"):
        safety.validate(order)


def test_allowlist_normalizes_dash_symbol_format() -> None:
    safety = BybitExecutionSafety.from_values(allowlist_symbols=["BTC-USDT"])
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0)

    safety.validate(order)


def test_max_order_notional_rejects_oversized_order() -> None:
    safety = BybitExecutionSafety(max_order_notional=999.0)
    order = Order("BTC/USDT", OrderSide.BUY, amount=10.0, price=100.0)

    with pytest.raises(BybitExecutionSafetyError, match="max_order_notional"):
        safety.validate(order)


def test_private_sync_requirement_rejects_when_not_synced() -> None:
    safety = BybitExecutionSafety(require_private_sync=True)
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0)

    with pytest.raises(BybitExecutionSafetyError, match="not synced"):
        safety.validate(order, private_synced=False)

    safety.validate(order, private_synced=True)
