from __future__ import annotations

from core.config.storage_config import StorageSettings
from core.interfaces.event_repository import EventRepository
from application.event_journal import EventJournal
from infrastructure.repositories.sqlite_event_repository import SQLiteEventRepository
from infrastructure.repositories.supabase_event_repository import SupabaseEventRepository
from infrastructure.storage.sqlite_connection import SQLiteConnection
from infrastructure.storage.supabase_connection import SupabaseConnection


def build_event_repository(settings: StorageSettings) -> EventRepository:
    driver = settings.driver.strip().lower()
    if driver == "sqlite":
        repo = SQLiteEventRepository(
            SQLiteConnection(settings.sqlite_path),
            table=settings.events_table,
        )
        repo.ensure_schema()
        return repo
    if driver == "supabase":
        return SupabaseEventRepository(
            SupabaseConnection(
                url=settings.supabase_url,
                service_key=settings.supabase_service_key,
                schema=settings.supabase_schema,
            ),
            table=settings.events_table,
        )
    raise ValueError(f"Unsupported storage driver: {settings.driver!r}")


def build_event_journal(settings: StorageSettings) -> EventJournal | None:
    if not settings.journal_enabled:
        return None
    return EventJournal(
        build_event_repository(settings),
        session_id=settings.journal_session_id or None,
        source=settings.journal_source,
    )
