from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageSettings:
    driver: str = "sqlite"
    sqlite_path: str = "data/trading_runtime.sqlite"
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_schema: str = "public"
    events_table: str = "trading_events"
    journal_enabled: bool = False
    journal_session_id: str = ""
    journal_source: str = "trading_runtime"
