import asyncio

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from infrastructure.repositories.supabase_order_intent_store import SupabaseOrderIntentStore
from infrastructure.storage.supabase_connection import SupabaseConnection


class FakeResponse:
    def __init__(self, payload=None) -> None:
        self._payload = payload
        self.content = b"[]" if payload is not None else b""

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def make_command(client_order_id: str = "client-1") -> PlaceOrderCommand:
    order = Order("BTC/USDT", OrderSide.BUY, amount=1.0, price=100.0, client_order_id=client_order_id)
    return PlaceOrderCommand(order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))


def make_store() -> SupabaseOrderIntentStore:
    return SupabaseOrderIntentStore(
        SupabaseConnection(url="https://example.supabase.co", service_key="sb_secret_test", schema="public")
    )


def test_supabase_order_intent_store_records_pending_with_ignore_duplicates(monkeypatch) -> None:
    calls = []

    def fake_post(url, *, headers=None, params=None, json=None, timeout=None):
        calls.append((url, headers, params, json, timeout))
        return FakeResponse([json])

    monkeypatch.setattr("requests.post", fake_post)
    store = make_store()

    recorded = asyncio.run(store.record_pending(make_command()))

    assert recorded is True
    url, headers, params, body, timeout = calls[0]
    assert url == "https://example.supabase.co/rest/v1/order_intents"
    assert headers["Prefer"] == "resolution=ignore-duplicates,return=representation"
    assert params == {"on_conflict": "client_order_id"}
    assert body["client_order_id"] == "client-1"
    assert body["status"] == "pending"


def test_supabase_order_intent_store_duplicate_returns_false(monkeypatch) -> None:
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse([]))
    store = make_store()

    assert asyncio.run(store.record_pending(make_command())) is False


def test_supabase_order_intent_store_marks_and_reads(monkeypatch) -> None:
    patch_calls = []

    def fake_patch(url, *, headers=None, params=None, json=None, timeout=None):
        patch_calls.append((url, params, json))
        return FakeResponse()

    def fake_get(url, *, headers=None, params=None, timeout=None):
        return FakeResponse(
            [
                {
                    "client_order_id": "client-1",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "status": "accepted",
                    "reason": "ENTRY",
                    "order_id": "order-1",
                    "reject_reason": None,
                    "ts": "2024-01-01T00:00:00",
                }
            ]
        )

    monkeypatch.setattr("requests.patch", fake_patch)
    monkeypatch.setattr("requests.get", fake_get)
    store = make_store()

    asyncio.run(store.mark_accepted("client-1", "order-1"))
    asyncio.run(store.mark_rejected("client-2", "min qty"))
    intent = asyncio.run(store.get("client-1"))

    assert patch_calls[0][1] == {"client_order_id": "eq.client-1"}
    assert patch_calls[0][2]["status"] == "accepted"
    assert patch_calls[0][2]["order_id"] == "order-1"
    assert patch_calls[1][1] == {"client_order_id": "eq.client-2"}
    assert patch_calls[1][2]["status"] == "rejected"
    assert patch_calls[1][2]["reject_reason"] == "min qty"
    assert intent.status == "accepted"
    assert intent.order_id == "order-1"
