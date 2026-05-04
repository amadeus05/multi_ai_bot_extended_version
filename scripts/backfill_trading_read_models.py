from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide, OrderType
from core.types.events import FillEvent, OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent
from infrastructure.repositories.sqlite_trading_read_model_repository import SQLiteTradingReadModelRepository
from infrastructure.storage.sqlite_connection import SQLiteConnection


def _order(payload: dict | None) -> Order | None:
    if not payload:
        return None
    return Order(
        symbol=str(payload["symbol"]),
        side=OrderSide(str(payload["side"])),
        amount=float(payload["amount"]),
        price=float(payload["price"]) if payload.get("price") is not None else None,
        type=OrderType(str(payload.get("type", "market"))),
        id=payload.get("id"),
        client_order_id=payload.get("client_order_id"),
        meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
    )


def _command(kind: str, payload: dict):
    if kind == "PlaceOrderCommand":
        order = _order(payload.get("order"))
        if order is None:
            return None
        return PlaceOrderCommand(
            order=order,
            reason=payload.get("reason"),
            ts=pd.Timestamp(payload["ts"]) if payload.get("ts") else None,
            continue_with_entry=bool(payload.get("continue_with_entry", False)),
        )
    if kind == "CancelOrderCommand":
        return CancelOrderCommand(
            order_id=str(payload["order_id"]),
            reason=payload.get("reason"),
            ts=pd.Timestamp(payload["ts"]) if payload.get("ts") else None,
        )
    return None


def _event(kind: str, payload: dict):
    if kind == "OrderAcceptedEvent":
        order = _order(payload.get("order"))
        if order is None:
            return None
        return OrderAcceptedEvent(
            ts=pd.Timestamp(payload["ts"]),
            order_id=str(payload["order_id"]),
            client_order_id=str(payload["client_order_id"]),
            order=order,
            event_id=payload.get("event_id"),
        )
    if kind == "OrderRejectedEvent":
        return OrderRejectedEvent(
            ts=pd.Timestamp(payload["ts"]),
            client_order_id=payload.get("client_order_id"),
            reason=str(payload.get("reason", "rejected")),
            order=_order(payload.get("order")),
            event_id=payload.get("event_id"),
        )
    if kind == "OrderCancelledEvent":
        return OrderCancelledEvent(
            ts=pd.Timestamp(payload["ts"]),
            order_id=str(payload["order_id"]),
            client_order_id=payload.get("client_order_id"),
            reason=payload.get("reason"),
            event_id=payload.get("event_id"),
        )
    if kind == "FillEvent":
        return FillEvent(
            ts=pd.Timestamp(payload["ts"]),
            order_id=str(payload["order_id"]),
            client_order_id=payload.get("client_order_id"),
            symbol=str(payload["symbol"]),
            side=OrderSide(str(payload["side"])),
            amount=float(payload["amount"]),
            price=float(payload["price"]),
            fee=float(payload.get("fee", 0.0)),
            meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
            command_reason=payload.get("command_reason"),
            continue_with_entry=bool(payload.get("continue_with_entry", False)),
            event_id=payload.get("event_id"),
        )
    return None


def _reset_projection_tables(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        for table in ("closed_trades", "active_trade_lots", "fills", "orders"):
            conn.execute(f"DELETE FROM {table}")


async def backfill(db_path: str, events_table: str, *, reset: bool) -> None:
    repo = SQLiteTradingReadModelRepository(SQLiteConnection(db_path))
    repo.ensure_schema()
    if reset:
        _reset_projection_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(
            conn.execute(
                f"""
                SELECT session_id, sequence, kind, payload_json
                FROM {events_table}
                ORDER BY session_id ASC, sequence ASC
                """
            )
        )

    projected = 0
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        command = _command(row["kind"], payload)
        if command is not None:
            await repo.record_command(row["session_id"], int(row["sequence"]), command)
            projected += 1
            continue
        event = _event(row["kind"], payload)
        if event is not None:
            await repo.record_event(row["session_id"], int(row["sequence"]), event)
            projected += 1

    print(f"Backfill complete | rows={len(rows)} | projected={projected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill dashboard read-model tables from trading_events.")
    parser.add_argument("--db", default="data/trading_runtime.sqlite", help="SQLite runtime database path.")
    parser.add_argument("--events-table", default="trading_events", help="Raw event journal table name.")
    parser.add_argument("--reset", action="store_true", help="Clear projection tables before replaying events.")
    args = parser.parse_args()
    asyncio.run(backfill(args.db, args.events_table, reset=args.reset))


if __name__ == "__main__":
    main()
