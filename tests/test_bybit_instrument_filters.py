import pytest

from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderType
from infrastructure.exchanges.bybit.bybit_instrument_filters import (
    BybitInstrumentFilter,
    BybitInstrumentFilterCache,
    BybitInstrumentFilterError,
)


def make_filter() -> BybitInstrumentFilter:
    return BybitInstrumentFilter.from_instrument(
        {
            "symbol": "BTCUSDT",
            "priceFilter": {"tickSize": "0.5"},
            "lotSizeFilter": {
                "minOrderQty": "0.001",
                "qtyStep": "0.001",
                "minNotionalValue": "5",
                "maxOrderQty": "100",
                "maxMktOrderQty": "50",
            },
        }
    )


def test_filter_parses_instruments_info_row() -> None:
    instrument_filter = make_filter()

    assert instrument_filter.symbol == "BTCUSDT"
    assert str(instrument_filter.tick_size) == "0.5"
    assert str(instrument_filter.qty_step) == "0.001"
    assert str(instrument_filter.min_qty) == "0.001"
    assert str(instrument_filter.min_notional) == "5"
    assert str(instrument_filter.max_limit_qty) == "100"
    assert str(instrument_filter.max_market_qty) == "50"


def test_normalize_order_rounds_qty_down_and_buy_price_up() -> None:
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.123456, price=25000.21, type=OrderType.LIMIT)

    make_filter().normalize_order(order)

    assert order.amount == pytest.approx(0.123)
    assert order.price == pytest.approx(25000.5)


def test_normalize_order_rounds_sell_price_down() -> None:
    order = Order("BTC/USDT", OrderSide.SELL, amount=0.123999, price=25000.49, type=OrderType.LIMIT)

    make_filter().normalize_order(order)

    assert order.amount == pytest.approx(0.123)
    assert order.price == pytest.approx(25000.0)


def test_rejects_below_min_qty_after_rounding() -> None:
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.0009, price=25000.0, type=OrderType.LIMIT)

    with pytest.raises(BybitInstrumentFilterError, match="below minOrderQty"):
        make_filter().normalize_order(order)


def test_rejects_below_min_notional_when_price_available() -> None:
    order = Order("BTC/USDT", OrderSide.BUY, amount=0.001, price=100.0, type=OrderType.LIMIT)

    with pytest.raises(BybitInstrumentFilterError, match="below minNotionalValue"):
        make_filter().normalize_order(order)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def get(self, path, params, *, signed=True, request_name=None):
        self.calls.append((path, params, signed, request_name))
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "ETHUSDT",
                        "priceFilter": {"tickSize": "0.01"},
                        "lotSizeFilter": {"minOrderQty": "0.01", "qtyStep": "0.01", "minNotionalValue": "5"},
                    }
                ]
            },
        }


def test_cache_fetches_filter_with_unsigned_instruments_info_request() -> None:
    client = FakeClient()
    cache = BybitInstrumentFilterCache(client, category="linear")

    instrument_filter = cache.get("ETH/USDT")

    assert instrument_filter.symbol == "ETHUSDT"
    assert client.calls == [
        (
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": "ETHUSDT"},
            False,
            "instruments_info",
        )
    ]
