from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Position
from core.types.enums import OrderSide, OrderStatus, OrderType, PositionSide


@dataclass(frozen=True)
class BybitExecutionMapper:
    category: str = "linear"
    default_position_idx: int = 0

    def to_api_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "/").replace("/", "").upper()

    def from_api_symbol(self, symbol: str) -> str:
        upper = str(symbol).upper()
        if upper.endswith("USDT") and "/" not in upper:
            return f"{upper[:-4]}/USDT"
        return upper

    def to_create_order_payload(self, command: PlaceOrderCommand) -> dict[str, Any]:
        order = command.order
        payload: dict[str, Any] = {
            "category": self.category,
            "symbol": self.to_api_symbol(order.symbol),
            "side": self.to_bybit_side(order.side),
            "orderType": self.to_bybit_order_type(order.type),
            "qty": self._number_string(order.amount),
            "positionIdx": self._position_idx(order),
        }
        if order.client_order_id:
            payload["orderLinkId"] = order.client_order_id
        if self._reduce_only(command):
            payload["reduceOnly"] = True
        elif self._meta_bool(order, "reduceOnly") is False or self._meta_bool(order, "reduce_only") is False:
            payload["reduceOnly"] = False

        if order.type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError("Bybit limit order requires price")
            payload["price"] = self._number_string(order.price)
            payload["timeInForce"] = str((order.meta or {}).get("time_in_force", "GTC"))
        elif (order.meta or {}).get("time_in_force"):
            payload["timeInForce"] = str(order.meta["time_in_force"])

        return payload

    def to_cancel_order_payload(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"category": self.category}
        if symbol:
            payload["symbol"] = self.to_api_symbol(symbol)
        if order_id.startswith("link:"):
            payload["orderLinkId"] = order_id.removeprefix("link:")
        else:
            payload["orderId"] = order_id
        return payload

    def to_status(self, row: dict[str, Any]) -> dict[str, Any]:
        order_id = str(row.get("orderId") or "")
        client_order_id = str(row.get("orderLinkId") or "") or None
        status = self.to_order_status(str(row.get("orderStatus") or ""))
        return {
            "status": status,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": self.from_api_symbol(str(row.get("symbol") or "")),
            "side": self.from_bybit_side(str(row.get("side") or "")),
            "filled": self._float(row.get("cumExecQty")),
            "avg_price": self._float(row.get("avgPrice") or row.get("price")),
            "fee": self._fee(row),
            "reason": row.get("rejectReason") or row.get("cancelType") or "",
            "ts": self._timestamp(row.get("updatedTime") or row.get("createdTime")),
            "raw": dict(row),
        }

    def to_position(self, row: dict[str, Any]) -> Position | None:
        size = abs(self._float(row.get("size")))
        if size <= 0:
            return None
        side = PositionSide.LONG if str(row.get("side")).lower() == "buy" else PositionSide.SHORT
        return Position(
            symbol=self.from_api_symbol(str(row.get("symbol") or "")),
            side=side,
            amount=size,
            entry_price=self._float(row.get("avgPrice")),
            meta={"bybit": dict(row)},
        )

    @staticmethod
    def to_bybit_side(side: OrderSide) -> str:
        return "Buy" if side == OrderSide.BUY else "Sell"

    @staticmethod
    def from_bybit_side(side: str) -> OrderSide | None:
        value = side.lower()
        if value == "buy":
            return OrderSide.BUY
        if value == "sell":
            return OrderSide.SELL
        return None

    @staticmethod
    def to_bybit_order_type(order_type: OrderType) -> str:
        return "Limit" if order_type == OrderType.LIMIT else "Market"

    @staticmethod
    def to_order_status(status: str) -> OrderStatus:
        value = status.strip().lower()
        if value in {"filled"}:
            return OrderStatus.FILLED
        if value in {"cancelled", "canceled", "partiallyfilledcanceled"}:
            return OrderStatus.CANCELLED
        if value in {"rejected", "deactivated"}:
            return OrderStatus.REJECTED
        return OrderStatus.NEW

    @classmethod
    def _number_string(cls, value: float) -> str:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"Bybit order number must be positive and finite: {value}")
        return format(number, "f").rstrip("0").rstrip(".")

    def _position_idx(self, order: Order) -> int:
        meta = order.meta or {}
        if meta.get("positionIdx") is not None:
            return int(meta["positionIdx"])
        if meta.get("position_idx") is not None:
            return int(meta["position_idx"])
        return int(self.default_position_idx)

    def _reduce_only(self, command: PlaceOrderCommand) -> bool:
        order = command.order
        for key in ("reduceOnly", "reduce_only"):
            value = self._meta_bool(order, key)
            if value is not None:
                return value
        if command.reason is None:
            return False
        return str(command.reason).upper() != "ENTRY"

    @staticmethod
    def _meta_bool(order: Order, key: str) -> bool | None:
        meta = order.meta or {}
        if key not in meta:
            return None
        return bool(meta[key])

    @classmethod
    def _fee(cls, row: dict[str, Any]) -> float:
        if row.get("cumExecFee") not in (None, ""):
            return cls._float(row.get("cumExecFee"))
        fee_detail = row.get("cumFeeDetail")
        if isinstance(fee_detail, dict):
            return sum(cls._float(value) for value in fee_detail.values())
        return 0.0

    @staticmethod
    def _float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        return float(value)

    @staticmethod
    def _timestamp(value: Any) -> pd.Timestamp:
        if value in (None, ""):
            return pd.Timestamp.utcnow()
        return pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(None)
