import pandas as pd
import pytest

from core.types.enums import OrderSide, PositionSide
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from infrastructure.exchanges.bybit.bybit_execution_mapper import BybitExecutionMapper


def test_order_private_payload_maps_new_order_to_accepted_event() -> None:
    mapper = BybitExecutionMapper()
    payload = {
        "topic": "order",
        "creationTime": 1710000000000,
        "data": [
            {
                "orderId": "order-1",
                "orderLinkId": "client-1",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "qty": "0.1",
                "price": "30000",
                "updatedTime": "1710000000001",
            }
        ],
    }

    events = mapper.private_events_from_payload(payload)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, OrderAcceptedEvent)
    assert event.order_id == "order-1"
    assert event.client_order_id == "client-1"
    assert event.order.symbol == "BTC/USDT"
    assert event.order.side == OrderSide.BUY
    assert event.order.amount == pytest.approx(0.1)
    assert event.order.price == pytest.approx(30000.0)
    assert event.event_id == "order-1:accepted"


def test_order_private_payload_maps_cancelled_and_rejected_events() -> None:
    mapper = BybitExecutionMapper()
    payload = {
        "topic": "order",
        "creationTime": 1710000000000,
        "data": [
            {
                "orderId": "order-cancelled",
                "orderLinkId": "client-cancelled",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderStatus": "Cancelled",
                "cancelType": "CancelByUser",
            },
            {
                "orderId": "order-rejected",
                "orderLinkId": "client-rejected",
                "symbol": "ETHUSDT",
                "side": "Sell",
                "orderStatus": "Rejected",
                "rejectReason": "EC_OrderQtyTooLarge",
            },
        ],
    }

    events = mapper.private_events_from_payload(payload)

    assert len(events) == 2
    assert isinstance(events[0], OrderCancelledEvent)
    assert events[0].order_id == "order-cancelled"
    assert events[0].client_order_id == "client-cancelled"
    assert events[0].reason == "CancelByUser"
    assert isinstance(events[1], OrderRejectedEvent)
    assert events[1].client_order_id == "client-rejected"
    assert events[1].reason == "EC_OrderQtyTooLarge"


def test_order_filled_payload_is_ignored_because_execution_topic_owns_fills() -> None:
    mapper = BybitExecutionMapper()

    events = mapper.private_events_from_payload(
        {"topic": "order", "data": [{"orderId": "order-1", "orderStatus": "Filled"}]}
    )

    assert events == []


def test_execution_private_payload_maps_to_fill_event_with_stable_event_id() -> None:
    mapper = BybitExecutionMapper()
    payload = {
        "topic": "execution",
        "creationTime": 1710000000000,
        "data": [
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "orderId": "order-1",
                "orderLinkId": "client-1",
                "side": "Buy",
                "execId": "exec-1",
                "execPrice": "30001.5",
                "execQty": "0.05",
                "execFee": "0.12",
                "execType": "Trade",
                "execTime": "1710000000123",
                "seq": 123,
            }
        ],
    }

    events = mapper.private_events_from_payload(payload)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FillEvent)
    assert event.order_id == "order-1"
    assert event.client_order_id == "client-1"
    assert event.symbol == "BTC/USDT"
    assert event.side == OrderSide.BUY
    assert event.amount == pytest.approx(0.05)
    assert event.price == pytest.approx(30001.5)
    assert event.fee == pytest.approx(0.12)
    assert event.ts == pd.Timestamp("2024-03-09T16:00:00.123")
    assert event.event_id == "order-1:fill:exec-1"
    assert event.meta["exec_type"] == "Trade"
    assert event.meta["seq"] == 123


def test_position_private_payload_maps_to_position_snapshots() -> None:
    mapper = BybitExecutionMapper()
    payload = {
        "topic": "position",
        "data": [
            {
                "symbol": "BTCUSDT",
                "side": "Sell",
                "size": "0.25",
                "avgPrice": "31000",
            }
        ],
    }

    positions = mapper.private_positions_from_payload(payload)

    assert len(positions) == 1
    assert positions[0].symbol == "BTC/USDT"
    assert positions[0].side == PositionSide.SHORT
    assert positions[0].amount == pytest.approx(0.25)
    assert positions[0].entry_price == pytest.approx(31000.0)
