import pandas as pd
import pytest

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderStatus, OrderType
from infrastructure.exchanges.bybit.bybit_execution_mapper import BybitExecutionMapper


def test_maps_entry_limit_order_to_create_payload() -> None:
    mapper = BybitExecutionMapper(category="linear")
    order = Order(
        "BTC/USDT",
        OrderSide.BUY,
        amount=0.125,
        price=25000.0,
        type=OrderType.LIMIT,
        client_order_id="entry-BTCUSDT-1",
        meta={"position_idx": 0},
    )

    payload = mapper.to_create_order_payload(PlaceOrderCommand(order, reason="ENTRY"))

    assert payload == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.125",
        "positionIdx": 0,
        "orderLinkId": "entry-BTCUSDT-1",
        "price": "25000",
        "timeInForce": "GTC",
    }


def test_maps_exit_order_to_reduce_only_market_payload() -> None:
    mapper = BybitExecutionMapper(category="linear")
    order = Order(
        "ETH/USDT",
        OrderSide.SELL,
        amount=1.5,
        type=OrderType.MARKET,
        client_order_id="exit-ETHUSDT-1",
    )

    payload = mapper.to_create_order_payload(PlaceOrderCommand(order, reason="SL"))

    assert payload["symbol"] == "ETHUSDT"
    assert payload["side"] == "Sell"
    assert payload["orderType"] == "Market"
    assert payload["qty"] == "1.5"
    assert payload["orderLinkId"] == "exit-ETHUSDT-1"
    assert payload["positionIdx"] == 0
    assert payload["reduceOnly"] is True
    assert "price" not in payload


def test_limit_order_without_price_is_rejected_before_rest() -> None:
    mapper = BybitExecutionMapper()
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, type=OrderType.LIMIT)

    with pytest.raises(ValueError, match="requires price"):
        mapper.to_create_order_payload(PlaceOrderCommand(order))


def test_maps_order_status_row_to_internal_status_dict() -> None:
    mapper = BybitExecutionMapper()
    status = mapper.to_status(
        {
            "orderId": "bybit-order-1",
            "orderLinkId": "client-1",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderStatus": "Filled",
            "cumExecQty": "0.2",
            "avgPrice": "30100.5",
            "cumExecFee": "0.15",
            "updatedTime": "1710000000000",
        }
    )

    assert status["status"] == OrderStatus.FILLED
    assert status["order_id"] == "bybit-order-1"
    assert status["client_order_id"] == "client-1"
    assert status["symbol"] == "BTC/USDT"
    assert status["side"] == OrderSide.BUY
    assert status["filled"] == pytest.approx(0.2)
    assert status["avg_price"] == pytest.approx(30100.5)
    assert status["fee"] == pytest.approx(0.15)
    assert status["ts"] == pd.Timestamp("2024-03-09T16:00:00")


def test_maps_bybit_order_status_names() -> None:
    mapper = BybitExecutionMapper()

    assert mapper.to_order_status("New") == OrderStatus.NEW
    assert mapper.to_order_status("PartiallyFilled") == OrderStatus.NEW
    assert mapper.to_order_status("Filled") == OrderStatus.FILLED
    assert mapper.to_order_status("Cancelled") == OrderStatus.CANCELLED
    assert mapper.to_order_status("Rejected") == OrderStatus.REJECTED
