from pathlib import Path

from core.config.storage_config import StorageSettings
from infrastructure.repositories.sqlite_order_intent_store import SQLiteOrderIntentStore
from infrastructure.repositories.supabase_order_intent_store import SupabaseOrderIntentStore
from infrastructure.storage.storage_factory import build_order_intent_store


def test_build_order_intent_store_uses_sqlite_driver() -> None:
    db_path = Path(".temp_code") / "factory_order_intents.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()

    store = build_order_intent_store(
        StorageSettings(driver="sqlite", sqlite_path=str(db_path))
    )

    assert isinstance(store, SQLiteOrderIntentStore)


def test_build_order_intent_store_uses_supabase_driver() -> None:
    store = build_order_intent_store(
        StorageSettings(
            driver="supabase",
            supabase_url="https://example.supabase.co",
            supabase_service_key="sb_secret_test",
        )
    )

    assert isinstance(store, SupabaseOrderIntentStore)
