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
        conn.execute(
            """
            CREATE TABLE closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                qty REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                fee REAL NOT NULL,
                exit_reason TEXT,
                duration_minutes REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO closed_trades (
                trade_id, session_id, symbol, direction, entry_time, exit_time,
                entry_price, exit_price, qty, pnl, pnl_pct, fee, exit_reason, duration_minutes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "closed-1",
                "paper-session",
                "SOL/USDT",
                "LONG",
                "2024-01-01T00:00:00",
                "2024-01-01T02:00:00",
                100.0,
                98.0,
                1.0,
                -2.1,
                -0.021,
                0.1,
                "SL",
                120.0,
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
    assert len(portfolio.closed_trade_results) == 1
    assert portfolio.closed_trade_results[0]["reason"] == "SL"
    assert portfolio._active_trade_number_by_symbol["ETH/USDT"] == 2
    assert portfolio._entry_fee_pool_by_symbol["ETH/USDT"] == pytest.approx(0.2)


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
        assert params["session_id"] == "eq.paper-main"
        if url == "https://example.supabase.co/rest/v1/active_trade_lots":
            assert params["remaining_qty"] == "gt.0.000000000001"
            return FakeResponse(
                [
                    {
                        "symbol": "ETH/USDT",
                        "direction": "long",
                        "remaining_qty": 0.1,
                        "entry_price": 2000.0,
                        "entry_ts": "2024-01-01T00:00:00",
                        "entry_fee_remaining": 0.1,
                        "meta_json": {"p_long": 0.7},
                    },
                    {
                        "symbol": "ETH/USDT",
                        "direction": "long",
                        "remaining_qty": 0.2,
                        "entry_price": 2100.0,
                        "entry_ts": "2024-01-01T01:00:00",
                        "entry_fee_remaining": 0.2,
                        "meta_json": {"p_long": 0.8},
                    },
                ]
            )
        if url == "https://example.supabase.co/rest/v1/closed_trades":
            return FakeResponse(
                [
                    {
                        "id": 1,
                        "trade_id": "closed-1",
                        "symbol": "DOGE/USDT",
                        "direction": "LONG",
                        "entry_time": "2024-01-01T00:00:00",
                        "exit_time": "2024-01-01T02:00:00",
                        "entry_price": 0.1,
                        "exit_price": 0.09,
                        "qty": 100.0,
                        "pnl": -1.0,
                        "pnl_pct": -0.1,
                        "fee": 0.01,
                        "exit_reason": "SL",
                        "duration_minutes": 120.0,
                    }
                ]
            )
        raise AssertionError(f"unexpected url: {url}")

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
    assert portfolio.closed_trade_results[0]["reason"] == "SL"
    assert portfolio._active_trade_number_by_symbol["ETH/USDT"] == 2
    assert portfolio._entry_fee_pool_by_symbol["ETH/USDT"] == pytest.approx(0.3)


class FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload
