from __future__ import annotations

import asyncio

import pandas as pd

from core.types.commands import PlaceOrderCommand
from domain.execution.order_intent_store import OrderIntent
from infrastructure.storage.sqlite_connection import SQLiteConnection


def _ts(value) -> str:
    return pd.Timestamp(value or pd.Timestamp.utcnow()).isoformat()


class SQLiteOrderIntentStore:
    def __init__(self, connection: SQLiteConnection) -> None:
        self._connection = connection

    def ensure_schema(self) -> None:
        with self._connection.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    client_order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    order_id TEXT,
                    reject_reason TEXT,
                    ts TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_order_intents_status
                ON order_intents(status, updated_at)
                """
            )

    async def record_pending(self, command: PlaceOrderCommand) -> bool:
        return await asyncio.to_thread(self._record_pending_sync, command)

    async def mark_accepted(self, client_order_id: str, order_id: str) -> None:
        await asyncio.to_thread(self._mark_accepted_sync, client_order_id, order_id)

    async def mark_rejected(self, client_order_id: str, reason: str) -> None:
        await asyncio.to_thread(self._mark_rejected_sync, client_order_id, reason)

    async def mark_cancelled(self, client_order_id: str, reason: str | None = None) -> None:
        await asyncio.to_thread(self._mark_cancelled_sync, client_order_id, reason)

    async def mark_filled(self, client_order_id: str, order_id: str | None = None) -> None:
        await asyncio.to_thread(self._mark_filled_sync, client_order_id, order_id)

    async def get(self, client_order_id: str) -> OrderIntent | None:
        return await asyncio.to_thread(self._get_sync, client_order_id)

    async def list_active(self) -> list[OrderIntent]:
        return await asyncio.to_thread(self._list_active_sync)

    def _record_pending_sync(self, command: PlaceOrderCommand) -> bool:
        self.ensure_schema()
        client_order_id = command.order.client_order_id
        if not client_order_id:
            raise ValueError("client_order_id is required before recording order intent")
        now = _ts(pd.Timestamp.utcnow())
        with self._connection.connect() as conn:
            result = conn.execute(
                """
                INSERT OR IGNORE INTO order_intents (
                    client_order_id, symbol, side, status, reason, ts, updated_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    client_order_id,
                    command.order.symbol,
                    command.order.side.value,
                    command.reason,
                    _ts(command.ts),
                    now,
                ),
            )
            return result.rowcount > 0

    def _mark_accepted_sync(self, client_order_id: str, order_id: str) -> None:
        self.ensure_schema()
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE order_intents
                SET status = 'accepted', order_id = ?, updated_at = ?
                WHERE client_order_id = ?
                """,
                (order_id, _ts(pd.Timestamp.utcnow()), client_order_id),
            )

    def _mark_rejected_sync(self, client_order_id: str, reason: str) -> None:
        self.ensure_schema()
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE order_intents
                SET status = 'rejected', reject_reason = ?, updated_at = ?
                WHERE client_order_id = ?
                """,
                (reason, _ts(pd.Timestamp.utcnow()), client_order_id),
            )

    def _mark_cancelled_sync(self, client_order_id: str, reason: str | None = None) -> None:
        self.ensure_schema()
        with self._connection.connect() as conn:
            conn.execute(
                """
                UPDATE order_intents
                SET status = 'cancelled', reject_reason = ?, updated_at = ?
                WHERE client_order_id = ?
                """,
                (reason, _ts(pd.Timestamp.utcnow()), client_order_id),
            )

    def _mark_filled_sync(self, client_order_id: str, order_id: str | None = None) -> None:
        self.ensure_schema()
        if order_id:
            sql = """
                UPDATE order_intents
                SET status = 'filled', order_id = ?, updated_at = ?
                WHERE client_order_id = ?
                """
            params = (order_id, _ts(pd.Timestamp.utcnow()), client_order_id)
        else:
            sql = """
                UPDATE order_intents
                SET status = 'filled', updated_at = ?
                WHERE client_order_id = ?
                """
            params = (_ts(pd.Timestamp.utcnow()), client_order_id)
        with self._connection.connect() as conn:
            conn.execute(sql, params)

    def _get_sync(self, client_order_id: str) -> OrderIntent | None:
        self.ensure_schema()
        with self._connection.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM order_intents
                WHERE client_order_id = ?
                """,
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return OrderIntent(
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            status=row["status"],
            reason=row["reason"],
            order_id=row["order_id"],
            reject_reason=row["reject_reason"],
            ts=pd.Timestamp(row["ts"]) if row["ts"] else None,
        )

    def _list_active_sync(self) -> list[OrderIntent]:
        self.ensure_schema()
        with self._connection.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM order_intents
                WHERE status IN ('pending', 'accepted')
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return [
            OrderIntent(
                client_order_id=row["client_order_id"],
                symbol=row["symbol"],
                side=row["side"],
                status=row["status"],
                reason=row["reason"],
                order_id=row["order_id"],
                reject_reason=row["reject_reason"],
                ts=pd.Timestamp(row["ts"]) if row["ts"] else None,
            )
            for row in rows
        ]
