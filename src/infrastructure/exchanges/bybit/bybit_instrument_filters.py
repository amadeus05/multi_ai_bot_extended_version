from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, InvalidOperation
from typing import Any

from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderType


class BybitInstrumentFilterError(ValueError):
    pass


@dataclass(frozen=True)
class BybitInstrumentFilter:
    symbol: str
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal | None = None
    max_limit_qty: Decimal | None = None
    max_market_qty: Decimal | None = None

    @classmethod
    def from_instrument(cls, row: dict[str, Any]) -> "BybitInstrumentFilter":
        price_filter = row.get("priceFilter") or {}
        lot_filter = row.get("lotSizeFilter") or {}
        return cls(
            symbol=str(row.get("symbol") or ""),
            tick_size=_positive_decimal(price_filter.get("tickSize"), "tickSize"),
            qty_step=_positive_decimal(lot_filter.get("qtyStep"), "qtyStep"),
            min_qty=_positive_decimal(lot_filter.get("minOrderQty"), "minOrderQty"),
            min_notional=_optional_decimal(lot_filter.get("minNotionalValue") or lot_filter.get("minOrderAmt")),
            max_limit_qty=_optional_decimal(lot_filter.get("maxOrderQty") or lot_filter.get("maxLimitOrderQty")),
            max_market_qty=_optional_decimal(lot_filter.get("maxMktOrderQty") or lot_filter.get("maxMarketOrderQty")),
        )

    def normalize_order(self, order: Order) -> Order:
        original_qty = _decimal(order.amount)
        qty = self.round_qty_down(original_qty)
        if qty < self.min_qty:
            raise BybitInstrumentFilterError(
                f"{self.symbol} qty {qty} is below minOrderQty {self.min_qty}"
            )

        price: Decimal | None = None
        if order.price is not None:
            original_price = _decimal(order.price)
            price = self.round_price_for_side(original_price, order.side)
            order.price = float(price)
        elif order.type == OrderType.LIMIT:
            raise BybitInstrumentFilterError(f"{self.symbol} limit order requires price")

        max_qty = self.max_market_qty if order.type == OrderType.MARKET else self.max_limit_qty
        if max_qty is not None and qty > max_qty:
            raise BybitInstrumentFilterError(f"{self.symbol} qty {qty} exceeds max order qty {max_qty}")

        notional_price = price
        if notional_price is None and order.price is not None:
            notional_price = _decimal(order.price)
        if self.min_notional is not None and notional_price is not None:
            notional = qty * notional_price
            if notional < self.min_notional:
                raise BybitInstrumentFilterError(
                    f"{self.symbol} notional {notional} is below minNotionalValue {self.min_notional}"
                )

        order.amount = float(qty)
        return order

    def round_price_for_side(self, price: Decimal, side: OrderSide) -> Decimal:
        rounding = ROUND_CEILING if side == OrderSide.BUY else ROUND_FLOOR
        return _round_to_step(price, self.tick_size, rounding)

    def round_qty_down(self, qty: Decimal) -> Decimal:
        return _round_to_step(qty, self.qty_step, ROUND_FLOOR)


class BybitInstrumentFilterCache:
    def __init__(self, client, *, category: str = "linear") -> None:
        self._client = client
        self._category = category
        self._filters: dict[str, BybitInstrumentFilter] = {}

    def set_filter(self, instrument_filter: BybitInstrumentFilter) -> None:
        self._filters[instrument_filter.symbol] = instrument_filter

    def get(self, symbol: str) -> BybitInstrumentFilter:
        api_symbol = _to_api_symbol(symbol)
        instrument_filter = self._filters.get(api_symbol)
        if instrument_filter is None:
            instrument_filter = self.fetch(api_symbol)
        return instrument_filter

    def fetch(self, symbol: str) -> BybitInstrumentFilter:
        api_symbol = _to_api_symbol(symbol)
        response = self._client.get(
            "/v5/market/instruments-info",
            {"category": self._category, "symbol": api_symbol},
            signed=False,
            request_name="instruments_info",
        )
        rows = (response.get("result") or {}).get("list") or []
        for row in rows:
            if str(row.get("symbol") or "").upper() == api_symbol:
                instrument_filter = BybitInstrumentFilter.from_instrument(row)
                self.set_filter(instrument_filter)
                return instrument_filter
        raise BybitInstrumentFilterError(f"Bybit instrument filter not found for {api_symbol}")

    def normalize_order(self, order: Order) -> Order:
        return self.get(order.symbol).normalize_order(order)


def _round_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if value <= 0:
        raise BybitInstrumentFilterError(f"value must be positive: {value}")
    units = (value / step).to_integral_value(rounding=rounding)
    rounded = units * step
    return rounded.normalize()


def _decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BybitInstrumentFilterError(f"invalid decimal value: {value}") from exc
    if not number.is_finite() or number <= 0:
        raise BybitInstrumentFilterError(f"decimal value must be positive and finite: {value}")
    return number


def _positive_decimal(value: Any, field_name: str) -> Decimal:
    if value in (None, ""):
        raise BybitInstrumentFilterError(f"missing {field_name}")
    return _decimal(value)


def _optional_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return _decimal(value)


def _to_api_symbol(symbol: str) -> str:
    return str(symbol).replace("-", "/").replace("/", "").upper()
