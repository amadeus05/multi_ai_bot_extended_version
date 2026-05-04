from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pandas as pd
import requests

from core.types.commands import CancelOrderCommand, PlaceOrderCommand, TradingCommand
from core.types.enums import OrderSide
from core.types.events import (
    FillEvent,
    OrderAcceptedEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
    TradingEvent,
)
from infrastructure.storage.supabase_connection import SupabaseConnection


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_value(payload: dict[str, Any] | None) -> dict[str, Any]:
    return json.loads(json.dumps(payload or {}, default=_json_default, ensure_ascii=False))


def _ts(value) -> str:
    return pd.Timestamp(value).isoformat()


def _meta_value(meta: dict[str, Any] | None, key: str, default=None):
    return (meta or {}).get(key, default)


def _duration_minutes(start, end) -> float:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)
    return max(0.0, (end_ts - start_ts).total_seconds() / 60.0)


class SupabaseTradingReadModelRepository:
    """Supabase/PostgREST projection repository for dashboard read models."""

    def __init__(self, connection: SupabaseConnection, *, timeout: float = 20.0) -> None:
        connection.validate_service_key()
        self._connection = connection
        self._timeout = float(timeout)

    async def record_command(self, session_id: str, sequence: int, command: TradingCommand) -> None:
        await asyncio.to_thread(self._record_command_sync, session_id, int(sequence), command)

    async def record_event(self, session_id: str, sequence: int, event: TradingEvent) -> None:
        await asyncio.to_thread(self._record_event_sync, session_id, int(sequence), event)

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        return self._connection.headers(prefer=prefer)

    def _request(self, method: str, table: str, *, params: dict | None = None, json_body=None, prefer: str | None = None):
        response = requests.request(
            method,
            self._connection.rest_url(table),
            headers=self._headers(prefer=prefer),
            params=params,
            json=json_body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return None

    def _upsert(self, table: str, row: dict[str, Any], *, on_conflict: str) -> None:
        self._request(
            "POST",
            table,
            params={"on_conflict": on_conflict},
            json_body=row,
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def _insert_ignore(self, table: str, row: dict[str, Any], *, on_conflict: str) -> bool:
        response = requests.post(
            self._connection.rest_url(table),
            headers=self._headers(prefer="resolution=ignore-duplicates,return=representation"),
            params={"on_conflict": on_conflict},
            json=row,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.content else []
        return bool(payload)

    def _patch(self, table: str, values: dict[str, Any], params: dict[str, str]) -> None:
        self._request("PATCH", table, params=params, json_body=values, prefer="return=minimal")

    def _delete(self, table: str, params: dict[str, str]) -> None:
        self._request("DELETE", table, params=params, prefer="return=minimal")

    def _select(self, table: str, params: dict[str, str]) -> list[dict]:
        payload = self._request("GET", table, params=params)
        return payload if isinstance(payload, list) else []

    def _record_command_sync(self, session_id: str, sequence: int, command: TradingCommand) -> None:
        if isinstance(command, PlaceOrderCommand):
            self._upsert_order_from_command(session_id, sequence, command)
        elif isinstance(command, CancelOrderCommand):
            self._mark_cancel_requested(session_id, sequence, command)

    def _record_event_sync(self, session_id: str, sequence: int, event: TradingEvent) -> None:
        if isinstance(event, OrderAcceptedEvent):
            self._upsert_order_from_accepted(session_id, sequence, event)
        elif isinstance(event, OrderRejectedEvent):
            self._upsert_order_from_rejected(session_id, sequence, event)
        elif isinstance(event, OrderCancelledEvent):
            self._mark_cancelled(session_id, sequence, event)
        elif isinstance(event, FillEvent):
            self._record_fill(session_id, sequence, event)

    def _order_conflict(self, row: dict[str, Any]) -> str:
        if row.get("client_order_id"):
            return "session_id,client_order_id"
        return "session_id,order_id"

    def _upsert_order_from_command(self, session_id: str, sequence: int, command: PlaceOrderCommand) -> None:
        order = command.order
        created_ts = _ts(command.ts or pd.Timestamp.utcnow())
        row = {
            "session_id": session_id,
            "sequence": sequence,
            "order_id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.type.value,
            "status": "submitted",
            "reason": command.reason,
            "amount": float(order.amount),
            "price": float(order.price) if order.price is not None else None,
            "created_ts": created_ts,
            "updated_ts": created_ts,
            "source": "command",
            "meta_json": _json_value(order.meta),
        }
        self._upsert("orders", row, on_conflict=self._order_conflict(row))

    def _upsert_order_from_accepted(self, session_id: str, sequence: int, event: OrderAcceptedEvent) -> None:
        order = event.order
        row = {
            "session_id": session_id,
            "sequence": sequence,
            "order_id": event.order_id,
            "client_order_id": event.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.type.value,
            "status": "accepted",
            "reason": None,
            "amount": float(order.amount),
            "price": float(order.price) if order.price is not None else None,
            "created_ts": _ts(event.ts),
            "updated_ts": _ts(event.ts),
            "source": "execution",
            "meta_json": _json_value(order.meta),
        }
        self._upsert("orders", row, on_conflict=self._order_conflict(row))

    def _upsert_order_from_rejected(self, session_id: str, sequence: int, event: OrderRejectedEvent) -> None:
        if event.order is None:
            return
        self._upsert_order_from_command(
            session_id,
            sequence,
            PlaceOrderCommand(event.order, reason=event.reason, ts=event.ts),
        )
        params = {"session_id": f"eq.{session_id}"}
        if event.client_order_id:
            params["client_order_id"] = f"eq.{event.client_order_id}"
        elif event.order.id:
            params["order_id"] = f"eq.{event.order.id}"
        else:
            return
        self._patch("orders", {"status": "rejected", "reason": event.reason, "updated_ts": _ts(event.ts)}, params)

    def _mark_cancel_requested(self, session_id: str, sequence: int, command: CancelOrderCommand) -> None:
        self._patch(
            "orders",
            {
                "status": "cancel_requested",
                "reason": command.reason,
                "sequence": sequence,
                "updated_ts": _ts(command.ts or pd.Timestamp.utcnow()),
            },
            {"session_id": f"eq.{session_id}", "order_id": f"eq.{command.order_id}"},
        )

    def _mark_cancelled(self, session_id: str, sequence: int, event: OrderCancelledEvent) -> None:
        params = {"session_id": f"eq.{session_id}"}
        if event.order_id:
            params["order_id"] = f"eq.{event.order_id}"
        elif event.client_order_id:
            params["client_order_id"] = f"eq.{event.client_order_id}"
        self._patch(
            "orders",
            {"status": "cancelled", "reason": event.reason, "sequence": sequence, "updated_ts": _ts(event.ts)},
            params,
        )

    def _record_fill(self, session_id: str, sequence: int, event: FillEvent) -> None:
        fill_id = event.event_id or f"{event.order_id}:fill:{sequence}"
        row = {
            "session_id": session_id,
            "sequence": sequence,
            "fill_id": fill_id,
            "order_id": event.order_id,
            "client_order_id": event.client_order_id,
            "symbol": event.symbol,
            "side": event.side.value,
            "amount": float(event.amount),
            "price": float(event.price),
            "fee": float(event.fee),
            "ts": _ts(event.ts),
            "command_reason": event.command_reason,
            "meta_json": _json_value(event.meta),
        }
        inserted = self._insert_ignore("fills", row, on_conflict="session_id,fill_id")
        self._patch(
            "orders",
            {"status": "filled", "updated_ts": _ts(event.ts)},
            {"session_id": f"eq.{session_id}", "order_id": f"eq.{event.order_id}"},
        )
        if inserted:
            self._project_fill_to_trades(session_id, fill_id, event)

    def _project_fill_to_trades(self, session_id: str, fill_id: str, event: FillEvent) -> None:
        qty_remaining = float(event.amount)
        if qty_remaining <= 0:
            return

        incoming_direction = "long" if event.side == OrderSide.BUY else "short"
        opposite_direction = "short" if incoming_direction == "long" else "long"
        lots = self._select(
            "active_trade_lots",
            {
                "session_id": f"eq.{session_id}",
                "symbol": f"eq.{event.symbol}",
                "direction": f"eq.{opposite_direction}",
                "select": "*",
                "order": "id.asc",
            },
        )

        exit_fee_remaining = float(event.fee)
        for lot in lots:
            if qty_remaining <= 1e-12:
                break
            lot_qty = float(lot["remaining_qty"])
            close_qty = min(qty_remaining, lot_qty)
            close_ratio = close_qty / lot_qty if lot_qty > 0 else 0.0
            fill_ratio = close_qty / float(event.amount) if float(event.amount) > 0 else 0.0
            entry_fee = float(lot["entry_fee_remaining"]) * close_ratio
            exit_fee = float(event.fee) * fill_ratio
            fee = entry_fee + exit_fee
            entry_price = float(lot["entry_price"])
            exit_price = float(event.price)
            gross = (
                (exit_price - entry_price) * close_qty
                if lot["direction"] == "long"
                else (entry_price - exit_price) * close_qty
            )
            notional = entry_price * close_qty
            pnl = gross - fee
            pnl_pct = pnl / notional if notional > 0 else 0.0
            duration_minutes = _duration_minutes(lot["entry_ts"], event.ts)
            lot_meta = lot.get("meta_json") if isinstance(lot.get("meta_json"), dict) else {}
            meta = {**lot_meta, **(event.meta or {})}
            trade_id = f"{lot['entry_fill_id']}:{fill_id}:{close_qty:.12f}"
            self._insert_ignore(
                "closed_trades",
                {
                    "trade_id": trade_id,
                    "session_id": session_id,
                    "symbol": event.symbol,
                    "direction": str(lot["direction"]).upper(),
                    "entry_order_id": lot["entry_order_id"],
                    "exit_order_id": event.order_id,
                    "entry_fill_id": lot["entry_fill_id"],
                    "exit_fill_id": fill_id,
                    "entry_time": lot["entry_ts"],
                    "exit_time": _ts(event.ts),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": close_qty,
                    "notional": notional,
                    "pnl_abs_gross": gross,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "fee": fee,
                    "exit_reason": event.command_reason or _meta_value(event.meta, "_engine_exit_reason") or _meta_value(event.meta, "reason"),
                    "duration_minutes": duration_minutes,
                    "p_long": _meta_value(meta, "p_long"),
                    "p_short": _meta_value(meta, "p_short"),
                    "signal_gap": _meta_value(meta, "signal_gap"),
                    "risk_pct": _meta_value(meta, "risk_pct"),
                    "meta_json": _json_value(meta),
                },
                on_conflict="session_id,trade_id",
            )
            new_lot_qty = lot_qty - close_qty
            new_entry_fee = max(0.0, float(lot["entry_fee_remaining"]) - entry_fee)
            new_notional = max(0.0, float(lot["entry_notional_remaining"]) - notional)
            if new_lot_qty <= 1e-12:
                self._delete("active_trade_lots", {"id": f"eq.{lot['id']}"})
            else:
                self._patch(
                    "active_trade_lots",
                    {
                        "remaining_qty": new_lot_qty,
                        "entry_fee_remaining": new_entry_fee,
                        "entry_notional_remaining": new_notional,
                    },
                    {"id": f"eq.{lot['id']}"},
                )
            qty_remaining -= close_qty
            exit_fee_remaining = max(0.0, exit_fee_remaining - exit_fee)

        if qty_remaining > 1e-12:
            self._insert_active_lot(
                session_id=session_id,
                fill_id=fill_id,
                event=event,
                direction=incoming_direction,
                qty=qty_remaining,
                fee=exit_fee_remaining,
            )

    def _insert_active_lot(self, *, session_id: str, fill_id: str, event: FillEvent, direction: str, qty: float, fee: float) -> None:
        self._request(
            "POST",
            "active_trade_lots",
            json_body={
                "session_id": session_id,
                "symbol": event.symbol,
                "direction": direction,
                "remaining_qty": float(qty),
                "entry_price": float(event.price),
                "entry_ts": _ts(event.ts),
                "entry_order_id": event.order_id,
                "entry_fill_id": fill_id,
                "entry_fee_remaining": float(fee),
                "entry_notional_remaining": float(qty) * float(event.price),
                "meta_json": _json_value(event.meta),
            },
            prefer="return=minimal",
        )
