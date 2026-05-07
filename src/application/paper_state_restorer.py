from __future__ import annotations

import asyncio
import json
import logging

import requests

from application.paper_warmup_catchup import PaperWarmupCatchup
from application.trading_engine import TradingEngine
from application.trading_state_restorer import RestoreResult
from core.config.storage_config import StorageSettings
from core.interfaces.data_provider import DataProvider
from core.types.domain_types import Position
from core.types.enums import PositionSide
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.storage.sqlite_connection import SQLiteConnection
from infrastructure.storage.supabase_connection import SupabaseConnection

logger = logging.getLogger(__name__)


class PaperDbStateRestorer:
    """Restore paper positions from the sqlite read model for a fixed journal session."""

    def __init__(
        self,
        *,
        portfolio: PortfolioManager,
        storage: StorageSettings,
    ) -> None:
        self._portfolio = portfolio
        self._storage = storage

    async def restore_trading_state(self) -> RestoreResult:
        if not self._storage.journal_enabled:
            return self._skipped("paper restore skipped: event journal disabled")
        if not self._storage.journal_session_id:
            return self._skipped("paper restore skipped: EVENT_JOURNAL_SESSION_ID is empty")

        positions = await asyncio.to_thread(self._load_positions)
        self._portfolio.positions = positions

        logger.info(
            "Paper state restored | source=paper_db | session_id=%s | positions=%s",
            self._storage.journal_session_id,
            len(positions),
        )
        return RestoreResult(
            restored=True,
            source="paper_db",
            positions_count=len(positions),
            message="paper state restored from active_trade_lots",
        )

    def _skipped(self, message: str) -> RestoreResult:
        logger.info(message)
        return RestoreResult(restored=True, source="paper_db", message=message)

    def _load_positions(self) -> list[Position]:
        driver = self._storage.driver.strip().lower()
        if driver == "sqlite":
            rows = self._load_sqlite_rows()
        elif driver == "supabase":
            rows = self._load_supabase_rows()
        else:
            logger.info("paper restore skipped: unsupported storage driver %r", self._storage.driver)
            rows = []

        positions: list[Position] = []
        for row in rows:
            direction = str(row["direction"]).lower()
            side = PositionSide.LONG if direction == "long" else PositionSide.SHORT
            meta = self._load_meta(row["meta_json"])
            meta.setdefault("restored_from", "paper_db")
            meta.setdefault("entry_ts", row["entry_ts"])
            positions.append(
                Position(
                    symbol=str(row["symbol"]),
                    side=side,
                    amount=float(row["amount"]),
                    entry_price=float(row["entry_price"]),
                    meta=meta,
                )
            )
        return positions

    def _load_sqlite_rows(self) -> list:
        connection = SQLiteConnection(self._storage.sqlite_path)
        with connection.connect() as conn:
            table_exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'active_trade_lots'
                """
            ).fetchone()
            if table_exists is None:
                return []

            return list(
                conn.execute(
                    """
                    SELECT
                        symbol,
                        direction,
                        SUM(remaining_qty) AS amount,
                        SUM(entry_notional_remaining) / NULLIF(SUM(remaining_qty), 0) AS entry_price,
                        MIN(entry_ts) AS entry_ts,
                        MAX(meta_json) AS meta_json
                    FROM active_trade_lots
                    WHERE session_id = ?
                    GROUP BY symbol, direction
                    HAVING SUM(remaining_qty) > 1e-12
                    """,
                    (self._storage.journal_session_id,),
                )
            )

    def _load_supabase_rows(self) -> list[dict]:
        connection = SupabaseConnection(
            url=self._storage.supabase_url,
            service_key=self._storage.supabase_service_key,
            schema=self._storage.supabase_schema,
        )
        connection.validate_service_key()
        response = requests.get(
            connection.rest_url("active_trade_lots"),
            headers=connection.headers(),
            params={
                "session_id": f"eq.{self._storage.journal_session_id}",
                "remaining_qty": "gt.0.000000000001",
                "select": "symbol,direction,remaining_qty,entry_price,entry_ts,meta_json",
                "order": "entry_ts.asc",
            },
            timeout=20,
        )
        response.raise_for_status()
        lots = response.json()
        grouped: dict[tuple[str, str], dict] = {}
        for lot in lots:
            key = (str(lot["symbol"]), str(lot["direction"]).lower())
            qty = float(lot["remaining_qty"])
            current = grouped.setdefault(
                key,
                {
                    "symbol": lot["symbol"],
                    "direction": lot["direction"],
                    "amount": 0.0,
                    "notional": 0.0,
                    "entry_ts": lot["entry_ts"],
                    "meta_json": lot.get("meta_json") or {},
                },
            )
            current["amount"] += qty
            current["notional"] += qty * float(lot["entry_price"])
            if str(lot["entry_ts"]) < str(current["entry_ts"]):
                current["entry_ts"] = lot["entry_ts"]
            if lot.get("meta_json"):
                current["meta_json"] = lot["meta_json"]
        return [
            {
                "symbol": value["symbol"],
                "direction": value["direction"],
                "amount": value["amount"],
                "entry_price": value["notional"] / value["amount"] if value["amount"] > 0 else 0.0,
                "entry_ts": value["entry_ts"],
                "meta_json": value["meta_json"],
            }
            for value in grouped.values()
            if float(value["amount"]) > 1e-12
        ]

    @staticmethod
    def _load_meta(raw) -> dict:
        if isinstance(raw, dict):
            return dict(raw)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to decode paper position meta_json")
            return {}
        return value if isinstance(value, dict) else {}


class PaperStateRestorer:
    """Paper-only restorer: DB positions first, then warmup catch-up exits."""

    def __init__(
        self,
        *,
        storage: StorageSettings,
        portfolio: PortfolioManager,
        data_provider: DataProvider,
        engine: TradingEngine,
        exit_manager: ExitManager,
        symbols: list[str],
        required_bars: int,
    ) -> None:
        self._storage = storage
        self._portfolio = portfolio
        self._data_provider = data_provider
        self._engine = engine
        self._exit_manager = exit_manager
        self._symbols = symbols
        self._required_bars = required_bars

    async def restore_trading_state(self) -> RestoreResult:
        result = await PaperDbStateRestorer(
            portfolio=self._portfolio,
            storage=self._storage,
        ).restore_trading_state()
        if not self._portfolio.get_open_positions():
            return result

        warmup_frames = {}
        for symbol in self._symbols:
            frame = await self._data_provider.warmup(symbol, self._required_bars)
            if not frame.empty:
                warmup_frames[symbol] = frame

        await PaperWarmupCatchup(
            portfolio=self._portfolio,
            engine=self._engine,
            exit_manager=self._exit_manager,
        ).run(warmup_frames)
        return result
