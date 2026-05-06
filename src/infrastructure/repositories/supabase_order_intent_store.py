from __future__ import annotations

import asyncio

import pandas as pd
import requests

from core.types.commands import PlaceOrderCommand
from domain.execution.order_intent_store import OrderIntent
from infrastructure.storage.supabase_connection import SupabaseConnection


def _ts(value) -> str:
    return pd.Timestamp(value or pd.Timestamp.utcnow()).isoformat()


class SupabaseOrderIntentStore:
    def __init__(self, connection: SupabaseConnection, *, table: str = "order_intents", timeout: float = 20.0) -> None:
        connection.validate_service_key()
        self._connection = connection
        self._table = table
        self._timeout = float(timeout)

    async def record_pending(self, command: PlaceOrderCommand) -> bool:
        return await asyncio.to_thread(self._record_pending_sync, command)

    async def mark_accepted(self, client_order_id: str, order_id: str) -> None:
        await asyncio.to_thread(
            self._patch,
            {"status": "accepted", "order_id": order_id, "updated_at": _ts(pd.Timestamp.utcnow())},
            {"client_order_id": f"eq.{client_order_id}"},
        )

    async def mark_rejected(self, client_order_id: str, reason: str) -> None:
        await asyncio.to_thread(
            self._patch,
            {"status": "rejected", "reject_reason": reason, "updated_at": _ts(pd.Timestamp.utcnow())},
            {"client_order_id": f"eq.{client_order_id}"},
        )

    async def get(self, client_order_id: str) -> OrderIntent | None:
        return await asyncio.to_thread(self._get_sync, client_order_id)

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        return self._connection.headers(prefer=prefer)

    def _record_pending_sync(self, command: PlaceOrderCommand) -> bool:
        client_order_id = command.order.client_order_id
        if not client_order_id:
            raise ValueError("client_order_id is required before recording order intent")
        row = {
            "client_order_id": client_order_id,
            "symbol": command.order.symbol,
            "side": command.order.side.value,
            "status": "pending",
            "reason": command.reason,
            "ts": _ts(command.ts),
            "updated_at": _ts(pd.Timestamp.utcnow()),
        }
        response = requests.post(
            self._connection.rest_url(self._table),
            headers=self._headers(prefer="resolution=ignore-duplicates,return=representation"),
            params={"on_conflict": "client_order_id"},
            json=row,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.content else []
        return bool(payload)

    def _patch(self, values: dict, params: dict[str, str]) -> None:
        response = requests.patch(
            self._connection.rest_url(self._table),
            headers=self._headers(prefer="return=minimal"),
            params=params,
            json=values,
            timeout=self._timeout,
        )
        response.raise_for_status()

    def _get_sync(self, client_order_id: str) -> OrderIntent | None:
        response = requests.get(
            self._connection.rest_url(self._table),
            headers=self._headers(),
            params={"client_order_id": f"eq.{client_order_id}", "select": "*", "limit": "1"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.content else []
        if not payload:
            return None
        row = payload[0]
        return OrderIntent(
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            status=row["status"],
            reason=row.get("reason"),
            order_id=row.get("order_id"),
            reject_reason=row.get("reject_reason"),
            ts=pd.Timestamp(row["ts"]) if row.get("ts") else None,
        )
