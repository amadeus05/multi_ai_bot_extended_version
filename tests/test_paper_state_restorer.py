import asyncio
from pathlib import Path

import pytest

from application.paper_state_restorer import PaperDbStateRestorer
from core.config.storage_config import StorageSettings
from core.types.enums import PositionSide
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.storage.sqlite_connection import SQLiteConnection


def test_paper_db_state_restorer_restores_active_lots() -> None:
    db_path = Path(".temp_code") / "test_paper_restore.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()
    with SQLiteConnection(str(db_path)).connect() as conn:
        conn.execute(
            """
            CREATE TABLE active_trade_lots (
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
            INSERT INTO active_trade_lots (
                session_id, symbol, direction, remaining_qty, entry_price, entry_ts,
                entry_order_id, entry_fill_id, entry_fee_remaining, entry_notional_remaining, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "paper-session",
                "ETH/USDT",
                "short",
                0.25,
                2000.0,
                "2024-01-01T00:00:00",
                "order-1",
                "fill-1",
                0.2,
                500.0,
                '{"p_short": 0.61}',
            ),
        )

    portfolio = PortfolioManager(cash={"USDT": 100.0})
    restorer = PaperDbStateRestorer(
        portfolio=portfolio,
        storage=StorageSettings(
            sqlite_path=str(db_path),
            journal_enabled=True,
            journal_session_id="paper-session",
        ),
    )

    result = asyncio.run(restorer.restore_trading_state())

    assert result.positions_count == 1
    assert portfolio.positions[0].symbol == "ETH/USDT"
    assert portfolio.positions[0].side == PositionSide.SHORT
    assert portfolio.positions[0].amount == 0.25
    assert portfolio.positions[0].entry_price == 2000.0
    assert portfolio.positions[0].meta["p_short"] == 0.61


def test_paper_db_state_restorer_skips_without_session_id() -> None:
    portfolio = PortfolioManager(cash={"USDT": 100.0})
    restorer = PaperDbStateRestorer(
        portfolio=portfolio,
        storage=StorageSettings(journal_enabled=True, journal_session_id=""),
    )

    result = asyncio.run(restorer.restore_trading_state())

    assert result.positions_count == 0
    assert "EVENT_JOURNAL_SESSION_ID is empty" in result.message
    assert portfolio.positions == []


def test_paper_db_state_restorer_restores_supabase_active_lots(monkeypatch) -> None:
    def fake_get(url, *, headers=None, params=None, timeout=None):
        assert url == "https://example.supabase.co/rest/v1/active_trade_lots"
        assert params["session_id"] == "eq.paper-main"
        assert params["remaining_qty"] == "gt.0.000000000001"
        return FakeResponse(
            [
                {
                    "symbol": "ETH/USDT",
                    "direction": "long",
                    "remaining_qty": 0.1,
                    "entry_price": 2000.0,
                    "entry_ts": "2024-01-01T00:00:00",
                    "meta_json": {"p_long": 0.7},
                },
                {
                    "symbol": "ETH/USDT",
                    "direction": "long",
                    "remaining_qty": 0.2,
                    "entry_price": 2100.0,
                    "entry_ts": "2024-01-01T01:00:00",
                    "meta_json": {"p_long": 0.8},
                },
            ]
        )

    monkeypatch.setattr("requests.get", fake_get)
    portfolio = PortfolioManager(cash={"USDT": 100.0})
    restorer = PaperDbStateRestorer(
        portfolio=portfolio,
        storage=StorageSettings(
            driver="supabase",
            supabase_url="https://example.supabase.co",
            supabase_service_key="sb_secret_test",
            journal_enabled=True,
            journal_session_id="paper-main",
        ),
    )

    result = asyncio.run(restorer.restore_trading_state())

    assert result.positions_count == 1
    assert portfolio.positions[0].symbol == "ETH/USDT"
    assert portfolio.positions[0].side == PositionSide.LONG
    assert portfolio.positions[0].amount == pytest.approx(0.3)
    assert portfolio.positions[0].entry_price == pytest.approx(2066.6666666666665)
    assert portfolio.positions[0].meta["entry_ts"] == "2024-01-01T00:00:00"


class FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload
