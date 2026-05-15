import asyncio
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from core.types.events import FillEvent, OrderAcceptedEvent
from infrastructure.repositories.sqlite_trading_read_model_repository import SQLiteTradingReadModelRepository
from infrastructure.storage.sqlite_connection import SQLiteConnection


def connect_rows(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def test_projection_records_orders_fills_and_closed_trades() -> None:
    db_path = Path(".temp_code") / "test_trading_read_model.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()
    repo = SQLiteTradingReadModelRepository(SQLiteConnection(str(db_path)))
    session_id = "test-session"

    entry_order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=100.0,
        client_order_id="entry-client",
        meta={"p_long": 0.72, "p_short": 0.28, "signal_gap": 0.44, "risk_pct": 1.0},
    )
    exit_order = Order(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=110.0,
        client_order_id="exit-client",
        meta={"reason": "TP"},
    )

    asyncio.run(repo.record_command(session_id, 1, PlaceOrderCommand(entry_order, reason="ENTRY", ts=pd.Timestamp("2024-01-01T00:00:00"))))
    asyncio.run(repo.record_event(session_id, 2, OrderAcceptedEvent(pd.Timestamp("2024-01-01T00:00:00"), "entry-order", "entry-client", entry_order)))
    asyncio.run(
        repo.record_event(
            session_id,
            3,
            FillEvent(
                ts=pd.Timestamp("2024-01-01T00:00:00"),
                order_id="entry-order",
                client_order_id="entry-client",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                amount=1.0,
                price=100.0,
                fee=0.1,
                meta=entry_order.meta,
                command_reason="ENTRY",
                event_id="entry-fill",
            ),
        )
    )
    asyncio.run(repo.record_command(session_id, 4, PlaceOrderCommand(exit_order, reason="TP", ts=pd.Timestamp("2024-01-01T02:00:00"))))
    asyncio.run(repo.record_event(session_id, 5, OrderAcceptedEvent(pd.Timestamp("2024-01-01T02:00:00"), "exit-order", "exit-client", exit_order)))
    asyncio.run(
        repo.record_event(
            session_id,
            6,
            FillEvent(
                ts=pd.Timestamp("2024-01-01T02:00:00"),
                order_id="exit-order",
                client_order_id="exit-client",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                amount=1.0,
                price=110.0,
                fee=0.1,
                meta={"reason": "TP"},
                command_reason="TP",
                event_id="exit-fill",
            ),
        )
    )

    with connect_rows(db_path) as con:
        assert con.execute("SELECT count(*) FROM orders").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM active_trade_lots").fetchone()[0] == 0
        trade = con.execute("SELECT * FROM closed_trades").fetchone()

    assert trade["symbol"] == "BTC/USDT"
    assert trade["direction"] == "LONG"
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["qty"] == pytest.approx(1.0)
    assert trade["pnl_abs_gross"] == pytest.approx(10.0)
    assert trade["fee"] == pytest.approx(0.2)
    assert trade["pnl"] == pytest.approx(9.8)
    assert trade["pnl_pct"] == pytest.approx(0.098)
    assert trade["exit_reason"] == "TP"
    assert trade["duration_minutes"] == pytest.approx(120.0)
    assert trade["p_long"] == pytest.approx(0.72)
    assert trade["signal_gap"] == pytest.approx(0.44)


def test_fill_updates_order_reason_from_command_reason() -> None:
    db_path = Path(".temp_code") / "test_trading_read_model_reason.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()
    repo = SQLiteTradingReadModelRepository(SQLiteConnection(str(db_path)))
    session_id = "test-session"
    order = Order(
        symbol="SOL/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=100.0,
        client_order_id="exit-client",
    )

    asyncio.run(repo.record_event(session_id, 1, OrderAcceptedEvent(pd.Timestamp("2024-01-01T00:00:00"), "exit-order", "exit-client", order)))
    asyncio.run(
        repo.record_event(
            session_id,
            2,
            FillEvent(
                ts=pd.Timestamp("2024-01-01T00:01:00"),
                order_id="exit-order",
                client_order_id="exit-client",
                symbol="SOL/USDT",
                side=OrderSide.SELL,
                amount=1.0,
                price=99.0,
                command_reason="SL",
                event_id="exit-fill",
            ),
        )
    )

    with connect_rows(db_path) as con:
        order_row = con.execute("SELECT status, reason FROM orders WHERE order_id = 'exit-order'").fetchone()

    assert order_row["status"] == "filled"
    assert order_row["reason"] == "SL"
