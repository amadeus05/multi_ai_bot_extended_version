import asyncio

import pandas as pd
import pytest

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderStatus, OrderType
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from infrastructure.exchanges.bybit.bybit_execution_exchange import BybitExecutionExchange
from infrastructure.exchanges.bybit.bybit_instrument_filters import BybitInstrumentFilter, BybitInstrumentFilterCache
from infrastructure.exchanges.bybit.bybit_rest_client import BybitRestError
from infrastructure.exchanges.bybit.bybit_execution_safety import BybitExecutionSafety


class FakeBybitClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict, str | None]] = []
        self.gets: list[tuple[str, dict, bool, str | None]] = []
        self.post_response = {"retCode": 0, "result": {"orderId": "bybit-order-1", "orderLinkId": "client-1"}}
        self.get_responses: list[dict] = []
        self.post_error: Exception | None = None

    def post(self, path: str, body: dict, *, request_name: str | None = None, signed: bool = True) -> dict:
        self.posts.append((path, body, request_name))
        if self.post_error is not None:
            raise self.post_error
        return self.post_response

    def get(self, path: str, params: dict, *, request_name: str | None = None, signed: bool = True) -> dict:
        self.gets.append((path, params, signed, request_name))
        return self.get_responses.pop(0)


def make_exchange(client: FakeBybitClient, *, safety: BybitExecutionSafety | None = None) -> BybitExecutionExchange:
    filters = BybitInstrumentFilterCache(client, category="linear")
    filters.set_filter(
        BybitInstrumentFilter.from_instrument(
            {
                "symbol": "BTCUSDT",
                "priceFilter": {"tickSize": "0.5"},
                "lotSizeFilter": {"minOrderQty": "0.001", "qtyStep": "0.001", "minNotionalValue": "5"},
            }
        )
    )
    filters.set_filter(
        BybitInstrumentFilter.from_instrument(
            {
                "symbol": "ETHUSDT",
                "priceFilter": {"tickSize": "0.01"},
                "lotSizeFilter": {"minOrderQty": "0.01", "qtyStep": "0.01", "minNotionalValue": "5"},
            }
        )
    )
    return BybitExecutionExchange(
        "key",
        "secret",
        testnet=True,
        client=client,
        instrument_filters=filters,
        safety=safety,
    )


def test_submit_order_lifecycle_returns_accepted_only_from_create_ack() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client)
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.1, price=25000.0, client_order_id="client-1")
    command = PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))

    events = asyncio.run(exchange.submit_order_lifecycle(command))

    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert not isinstance(events[0], FillEvent)
    assert events[0].order_id == "bybit-order-1"
    assert events[0].client_order_id == "client-1"
    assert order.id == "bybit-order-1"
    assert client.posts[0][0] == "/v5/order/create"
    assert client.posts[0][1]["orderLinkId"] == "client-1"


def test_submit_order_lifecycle_rounds_order_before_rest_submit() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client)
    order = Order(
        "BTC/USDT",
        OrderSide.BUY,
        amount=0.123456,
        price=25000.21,
        type=OrderType.LIMIT,
        client_order_id="client-1",
    )

    events = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))

    assert isinstance(events[0], OrderAcceptedEvent)
    assert client.posts[0][1]["qty"] == "0.123"
    assert client.posts[0][1]["price"] == "25000.5"


def test_submit_order_lifecycle_rejects_filter_failure_before_rest_submit() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client)
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.0009, price=25000.0, client_order_id="client-1")

    events = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))

    assert len(events) == 1
    assert isinstance(events[0], OrderRejectedEvent)
    assert "below minOrderQty" in events[0].reason
    assert client.posts == []


def test_submit_order_lifecycle_dry_run_returns_accepted_without_rest_submit() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client, safety=BybitExecutionSafety(dry_run=True))
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.1, price=25000.0, client_order_id="client-1")

    events = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))

    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert events[0].order_id == "dry-run:client-1"
    assert order.id == "dry-run:client-1"
    assert client.posts == []


def test_submit_order_lifecycle_rejects_safety_failure_before_rest_submit() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client, safety=BybitExecutionSafety(kill_switch=True))
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.1, price=25000.0, client_order_id="client-1")

    events = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))

    assert len(events) == 1
    assert isinstance(events[0], OrderRejectedEvent)
    assert "kill switch" in events[0].reason
    assert client.posts == []


def test_submit_order_lifecycle_requires_private_sync_when_configured() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client, safety=BybitExecutionSafety(require_private_sync=True))
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.1, price=25000.0, client_order_id="client-1")

    rejected = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))
    assert isinstance(rejected[0], OrderRejectedEvent)
    assert "not synced" in rejected[0].reason
    assert client.posts == []

    exchange.set_private_synced(True)
    accepted = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order, reason="ENTRY")))
    assert isinstance(accepted[0], OrderAcceptedEvent)
    assert len(client.posts) == 1


def test_submit_order_lifecycle_maps_bybit_rejection_to_rejected_event() -> None:
    client = FakeBybitClient()
    client.post_error = BybitRestError("order_create", "min qty", ret_code=110001, ret_msg="min qty")
    exchange = make_exchange(client)
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.1, client_order_id="client-1")

    events = asyncio.run(exchange.submit_order_lifecycle(PlaceOrderCommand(order)))

    assert len(events) == 1
    assert isinstance(events[0], OrderRejectedEvent)
    assert events[0].client_order_id == "client-1"
    assert "110001" in events[0].reason


def test_get_order_status_uses_realtime_then_history_fallback() -> None:
    client = FakeBybitClient()
    client.get_responses = [
        {"retCode": 0, "result": {"list": []}},
        {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "bybit-order-1",
                        "orderLinkId": "client-1",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderStatus": "Filled",
                        "cumExecQty": "1",
                        "avgPrice": "100",
                    }
                ]
            },
        },
    ]
    exchange = make_exchange(client)

    status = asyncio.run(exchange.get_order_status("bybit-order-1"))

    assert status["status"] == OrderStatus.FILLED
    assert [call[0] for call in client.gets] == ["/v5/order/realtime", "/v5/order/history"]


def test_cancel_order_lifecycle_emits_cancelled_event() -> None:
    client = FakeBybitClient()
    exchange = make_exchange(client)

    events = asyncio.run(exchange.cancel_order_lifecycle(CancelOrderCommand("bybit-order-1", reason="user")))

    assert len(events) == 1
    assert isinstance(events[0], OrderCancelledEvent)
    assert events[0].order_id == "bybit-order-1"
    assert client.posts[0][0] == "/v5/order/cancel"


def test_current_price_uses_unsigned_ticker_request() -> None:
    client = FakeBybitClient()
    client.get_responses = [{"retCode": 0, "result": {"list": [{"lastPrice": "101.5"}]}}]
    exchange = make_exchange(client)

    price = asyncio.run(exchange.current_price("BTC/USDT"))

    assert price == pytest.approx(101.5)
    assert client.gets[0] == ("/v5/market/tickers", {"category": "linear", "symbol": "BTCUSDT"}, False, "ticker")
