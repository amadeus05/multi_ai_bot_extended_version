import asyncio

import pandas as pd

from core.types.enums import OrderSide
from core.types.events import FillEvent
from infrastructure.repositories.supabase_trading_read_model_repository import SupabaseTradingReadModelRepository
from infrastructure.storage.supabase_connection import SupabaseConnection


def test_supabase_projection_marks_filled_order_with_fill_reason(monkeypatch) -> None:
    repo = SupabaseTradingReadModelRepository(
        SupabaseConnection(url="https://example.supabase.co", service_key="sb_secret_test", schema="public")
    )
    patch_calls = []

    monkeypatch.setattr(repo, "_insert_ignore", lambda *args, **kwargs: False)
    monkeypatch.setattr(repo, "_patch", lambda table, values, params: patch_calls.append((table, values, params)))

    asyncio.run(
        repo.record_event(
            "session-1",
            10,
            FillEvent(
                ts=pd.Timestamp("2024-01-01T00:00:00"),
                order_id="order-1",
                client_order_id="client-1",
                symbol="SOL/USDT",
                side=OrderSide.SELL,
                amount=1.0,
                price=100.0,
                command_reason="SL",
                event_id="fill-1",
            ),
        )
    )

    assert patch_calls == [
        (
            "orders",
            {"status": "filled", "updated_ts": "2024-01-01T00:00:00", "reason": "SL"},
            {"session_id": "eq.session-1", "order_id": "eq.order-1"},
        )
    ]
