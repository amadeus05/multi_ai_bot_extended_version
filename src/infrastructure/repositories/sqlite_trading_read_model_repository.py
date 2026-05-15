from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pandas as pd

from core.types.commands import CancelOrderCommand, PlaceOrderCommand, TradingCommand
from core.types.enums import OrderSide
from core.types.events import (
    FillEvent,
    OrderAcceptedEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
    TradingEvent,
)
from infrastructure.storage.sqlite_connection import SQLiteConnection


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


def _payload_json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, default=_json_default, ensure_ascii=False, sort_keys=True)


def _ts(value) -> str:
    return pd.Timestamp(value).isoformat()


def _meta_value(meta: dict[str, Any] | None, key: str, default=None):
    return (meta or {}).get(key, default)


def _fill_reason(event: FillEvent) -> str | None:
    return event.command_reason or _meta_value(event.meta, "_engine_exit_reason") or _meta_value(event.meta, "reason")


def _duration_minutes(start, end) -> float:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)
    return max(0.0, (end_ts - start_ts).total_seconds() / 60.0)


class SQLiteTradingReadModelRepository:
    """Projection tables for dashboard-friendly trading analytics.

    The raw event journal remains the audit source of truth. These tables are
    read models: orders/fills preserve execution facts, while closed_trades is
    a convenient aggregate built from fills using FIFO lots.
    """

    def __init__(self, connection: SQLiteConnection) -> None:
        self._connection = connection

    async def record_command(self, session_id: str, sequence: int, command: TradingCommand) -> None:
        await asyncio.to_thread(self._record_command_sync, session_id, int(sequence), command)

    async def record_event(self, session_id: str, sequence: int, event: TradingEvent) -> None:
        await asyncio.to_thread(self._record_event_sync, session_id, int(sequence), event)

    def ensure_schema(self) -> None:
        with self._connection.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    order_id TEXT,
                    client_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    amount REAL NOT NULL,
                    price REAL,
                    created_ts TEXT NOT NULL,
                    updated_ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    UNIQUE(session_id, order_id),
                    UNIQUE(session_id, client_order_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_orders_session_ts
                ON orders(session_id, updated_ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_orders_symbol_status
                ON orders(symbol, status)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    fill_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    client_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    ts TEXT NOT NULL,
                    command_reason TEXT,
                    meta_json TEXT NOT NULL,
                    UNIQUE(session_id, fill_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fills_session_ts
                ON fills(session_id, ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fills_order
                ON fills(session_id, order_id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_trade_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    remaining_qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_ts TEXT NOT NULL,
                    entry_order_id TEXT NOT NULL,
                    entry_fill_id TEXT NOT NULL,
                    entry_fee_remaining REAL NOT NULL,
                    entry_notional_remaining REAL NOT NULL,
                    meta_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_active_trade_lots_session_symbol
                ON active_trade_lots(session_id, symbol, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_order_id TEXT NOT NULL,
                    exit_order_id TEXT NOT NULL,
                    entry_fill_id TEXT NOT NULL,
                    exit_fill_id TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    notional REAL NOT NULL,
                    pnl_abs_gross REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    fee REAL NOT NULL,
                    exit_reason TEXT,
                    duration_minutes REAL NOT NULL,
                    p_long REAL,
                    p_short REAL,
                    signal_gap REAL,
                    risk_pct REAL,
                    meta_json TEXT NOT NULL,
                    UNIQUE(session_id, trade_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_closed_trades_session_exit
                ON closed_trades(session_id, exit_time)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_closed_trades_symbol
                ON closed_trades(symbol, exit_time)
                """
            )

    def _record_command_sync(self, session_id: str, sequence: int, command: TradingCommand) -> None:
        self.ensure_schema()
        if isinstance(command, PlaceOrderCommand):
            self._upsert_order_from_command(session_id, sequence, command)
        elif isinstance(command, CancelOrderCommand):
            self._mark_cancel_requested(session_id, sequence, command)

    def _record_event_sync(self, session_id: str, sequence: int, event: TradingEvent) -> None:
        self.ensure_schema()
        if isinstance(event, OrderAcceptedEvent):
            self._upsert_order_from_accepted(session_id, sequence, event)
        elif isinstance(event, OrderRejectedEvent):
            self._upsert_order_from_rejected(session_id, sequence, event)
        elif isinstance(event, OrderCancelledEvent):
            self._mark_cancelled(session_id, sequence, event)
        elif isinstance(event, FillEvent):
            self._record_fill(session_id, sequence, event)

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
            "meta_json": _payload_json(order.meta),
        }
        with self._connection.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    session_id, sequence, order_id, client_order_id, symbol, side,
                    order_type, status, reason, amount, price, created_ts,
                    updated_ts, source, meta_json
                )
                VALUES (
                    :session_id, :sequence, :order_id, :client_order_id, :symbol, :side,
                    :order_type, :status, :reason, :amount, :price, :created_ts,
                    :updated_ts, :source, :meta_json
                )
                ON CONFLICT(session_id, client_order_id) DO UPDATE SET
                    sequence=excluded.sequence,
                    order_id=COALESCE(excluded.order_id, orders.order_id),
                    status=excluded.status,
                    reason=excluded.reason,
                    amount=excluded.amount,
                    price=excluded.price,
                    updated_ts=excluded.updated_ts,
                    source=excluded.source,
                    meta_json=excluded.meta_json
                """,
                row,
            )

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
            "meta_json": _payload_json(order.meta),
        }
        with self._connection.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    session_id, sequence, order_id, client_order_id, symbol, side,
                    order_type, status, reason, amount, price, created_ts,
                    updated_ts, source, meta_json
                )
                VALUES (
                    :session_id, :sequence, :order_id, :client_order_id, :symbol, :side,
                    :order_type, :status, :reason, :amount, :price, :created_ts,
                    :updated_ts, :source, :meta_json
                )
                ON CONFLICT(session_id, client_order_id) DO UPDATE SET
                    sequence=excluded.sequence,
                    order_id=excluded.order_id,
                    status=excluded.status,
                    amount=excluded.amount,
                    price=excluded.price,
                    updated_ts=excluded.updated_ts,
                    source=excluded.source,
                    meta_json=excluded.meta_json
                """,
                row,
            )

    def _upsert_order_from_rejected(self, session_id: str, sequence: int, event: OrderRejectedEvent) -> None:
        if event.order is None:
            return
        command = PlaceOrderCommand(event.order, reason=event.reason, ts=event.ts)
        self._upsert_order_from_command(session_id, sequence, command)
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = 'rejected', reason = ?, updated_ts = ?
                WHERE session_id = ?
                  AND (client_order_id = ? OR order_id = ?)
                """,
                (event.reason, _ts(event.ts), session_id, event.client_order_id, event.order.id),
            )

    def _mark_cancel_requested(self, session_id: str, sequence: int, command: CancelOrderCommand) -> None:
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = 'cancel_requested', reason = ?, sequence = ?, updated_ts = ?
                WHERE session_id = ? AND order_id = ?
                """,
                (command.reason, sequence, _ts(command.ts or pd.Timestamp.utcnow()), session_id, command.order_id),
            )

    def _mark_cancelled(self, session_id: str, sequence: int, event: OrderCancelledEvent) -> None:
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = 'cancelled', reason = ?, sequence = ?, updated_ts = ?
                WHERE session_id = ?
                  AND (order_id = ? OR client_order_id = ?)
                """,
                (event.reason, sequence, _ts(event.ts), session_id, event.order_id, event.client_order_id),
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
            "meta_json": _payload_json(event.meta),
        }
        with self._connection.connect() as conn:
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO fills (
                    session_id, sequence, fill_id, order_id, client_order_id,
                    symbol, side, amount, price, fee, ts, command_reason, meta_json
                )
                VALUES (
                    :session_id, :sequence, :fill_id, :order_id, :client_order_id,
                    :symbol, :side, :amount, :price, :fee, :ts, :command_reason, :meta_json
                )
                """,
                row,
            ).rowcount
            conn.execute(
                """
                UPDATE orders
                SET status = 'filled',
                    reason = COALESCE(?, reason),
                    updated_ts = ?
                WHERE session_id = ?
                  AND (order_id = ? OR client_order_id = ?)
                """,
                (_fill_reason(event), _ts(event.ts), session_id, event.order_id, event.client_order_id),
            )
            if inserted:
                self._project_fill_to_trades(conn, session_id, fill_id, event)

    def _project_fill_to_trades(self, conn, session_id: str, fill_id: str, event: FillEvent) -> None:
        qty_remaining = float(event.amount)
        if qty_remaining <= 0:
            return

        incoming_direction = "long" if event.side == OrderSide.BUY else "short"
        opposite_direction = "short" if incoming_direction == "long" else "long"
        lots = list(
            conn.execute(
                """
                SELECT *
                FROM active_trade_lots
                WHERE session_id = ? AND symbol = ? AND direction = ?
                ORDER BY id ASC
                """,
                (session_id, event.symbol, opposite_direction),
            )
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
            lot_meta = json.loads(lot["meta_json"] or "{}")
            meta = {**lot_meta, **(event.meta or {})}
            trade_id = f"{lot['entry_fill_id']}:{fill_id}:{close_qty:.12f}"
            conn.execute(
                """
                INSERT OR IGNORE INTO closed_trades (
                    trade_id, session_id, symbol, direction, entry_order_id, exit_order_id,
                    entry_fill_id, exit_fill_id, entry_time, exit_time, entry_price,
                    exit_price, qty, notional, pnl_abs_gross, pnl, pnl_pct, fee,
                    exit_reason, duration_minutes, p_long, p_short, signal_gap, risk_pct,
                    meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    session_id,
                    event.symbol,
                    str(lot["direction"]).upper(),
                    lot["entry_order_id"],
                    event.order_id,
                    lot["entry_fill_id"],
                    fill_id,
                    lot["entry_ts"],
                    _ts(event.ts),
                    entry_price,
                    exit_price,
                    close_qty,
                    notional,
                    gross,
                    pnl,
                    pnl_pct,
                    fee,
                    _fill_reason(event),
                    duration_minutes,
                    _meta_value(meta, "p_long"),
                    _meta_value(meta, "p_short"),
                    _meta_value(meta, "signal_gap"),
                    _meta_value(meta, "risk_pct"),
                    _payload_json(meta),
                ),
            )
            new_lot_qty = lot_qty - close_qty
            new_entry_fee = max(0.0, float(lot["entry_fee_remaining"]) - entry_fee)
            new_notional = max(0.0, float(lot["entry_notional_remaining"]) - notional)
            if new_lot_qty <= 1e-12:
                conn.execute("DELETE FROM active_trade_lots WHERE id = ?", (lot["id"],))
            else:
                conn.execute(
                    """
                    UPDATE active_trade_lots
                    SET remaining_qty = ?, entry_fee_remaining = ?, entry_notional_remaining = ?
                    WHERE id = ?
                    """,
                    (new_lot_qty, new_entry_fee, new_notional, lot["id"]),
                )
            qty_remaining -= close_qty
            exit_fee_remaining = max(0.0, exit_fee_remaining - exit_fee)

        if qty_remaining > 1e-12:
            self._insert_active_lot(
                conn,
                session_id=session_id,
                fill_id=fill_id,
                event=event,
                direction=incoming_direction,
                qty=qty_remaining,
                fee=exit_fee_remaining,
            )

    @staticmethod
    def _insert_active_lot(conn, *, session_id: str, fill_id: str, event: FillEvent, direction: str, qty: float, fee: float) -> None:
        conn.execute(
            """
            INSERT INTO active_trade_lots (
                session_id, symbol, direction, remaining_qty, entry_price, entry_ts,
                entry_order_id, entry_fill_id, entry_fee_remaining,
                entry_notional_remaining, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event.symbol,
                direction,
                float(qty),
                float(event.price),
                _ts(event.ts),
                event.order_id,
                fill_id,
                float(fee),
                float(qty) * float(event.price),
                _payload_json(event.meta),
            ),
        )
