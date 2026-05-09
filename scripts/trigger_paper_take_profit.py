from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "runners"))

from application.event_journal import EventJournal
from application.trading_engine import ENGINE_META_EXIT_REASON, ENGINE_META_REASON_SUFFIX, TradingEngine
from core.config.loader import load_paper_settings, load_storage_settings
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order, Position, Tick
from core.types.enums import OrderSide, PositionSide
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from infrastructure.repositories.supabase_event_repository import SupabaseEventRepository
from infrastructure.repositories.supabase_trading_read_model_repository import SupabaseTradingReadModelRepository
from infrastructure.storage.storage_factory import build_order_intent_store
from infrastructure.storage.supabase_connection import SupabaseConnection


def _direction_to_side(direction: str) -> PositionSide:
    return PositionSide.LONG if direction.lower() == "long" else PositionSide.SHORT


def _exit_side(position: Position) -> OrderSide:
    return OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY


def _synthetic_tp_tick(position: Position, take_pct: float) -> Tick:
    ts = pd.Timestamp.now(tz="UTC").tz_convert(None)
    entry = float(position.entry_price)
    if position.side == PositionSide.LONG:
        high = entry * (1.0 + float(take_pct) * 1.25)
        low = entry
        close = high
    else:
        low = entry * (1.0 - float(take_pct) * 1.25)
        high = entry
        close = low
    return Tick(
        symbol=position.symbol,
        ts=ts,
        bid=close,
        ask=close,
        price=close,
        volume=0.0,
        open=entry,
        high=high,
        low=low,
        close=close,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger a synthetic TP close for an open paper position in Supabase.")
    parser.add_argument("--symbol", default="", help="Symbol to close, e.g. SOL/USDT. Defaults to the first open lot.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned close without writing to Supabase.")
    args = parser.parse_args()

    storage = load_storage_settings()
    settings = load_paper_settings()
    if storage.driver.strip().lower() != "supabase":
        raise RuntimeError("This smoke script requires STORAGE_DRIVER=supabase.")
    if not storage.journal_enabled or not storage.journal_session_id:
        raise RuntimeError("Set EVENT_JOURNAL_ENABLED=1 and EVENT_JOURNAL_SESSION_ID to the paper session id.")

    connection = SupabaseConnection(
        url=storage.supabase_url,
        service_key=storage.supabase_service_key,
        schema=storage.supabase_schema,
    )
    connection.validate_service_key()

    from application.paper_state_restorer import PaperDbStateRestorer

    portfolio = PortfolioManager(cash={"USDT": float(settings.trading.initial_capital)})
    await PaperDbStateRestorer(portfolio=portfolio, storage=storage).restore_trading_state()
    positions = portfolio.get_open_positions()
    if args.symbol:
        positions = [position for position in positions if position.symbol == args.symbol]
    if not positions:
        raise RuntimeError("No open paper positions found in active_trade_lots for the configured session.")

    position = positions[0]
    meta = position.meta or {}
    if meta.get("barrier_take_pct") is None or meta.get("barrier_stop_pct") is None:
        raise RuntimeError(f"Position {position.symbol} has no barrier_take_pct/barrier_stop_pct in meta_json.")

    trading = settings.trading
    exit_manager = ExitManager(slippage=float(trading.costs.slippage))
    tick = _synthetic_tp_tick(position, float(meta["barrier_take_pct"]))
    exit_price, reason = exit_manager.check_causal_exit(
        position=position,
        next_open=float(tick.open),
        next_high=float(tick.high),
        next_low=float(tick.low),
        stop_pct=float(meta["barrier_stop_pct"]),
        take_pct=float(meta["barrier_take_pct"]),
    )
    if exit_price is None:
        raise RuntimeError("Synthetic tick did not trigger an exit.")

    print(
        "Synthetic close:",
        {
            "symbol": position.symbol,
            "side": position.side.value,
            "amount": position.amount,
            "entry_price": position.entry_price,
            "take_pct": meta["barrier_take_pct"],
            "tick_high": tick.high,
            "tick_low": tick.low,
            "exit_price": exit_price,
            "reason": reason,
            "session_id": storage.journal_session_id,
        },
    )
    if args.dry_run:
        return

    exchange = SimulatedExchange(
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    execution = ExecutionService(order_intent_store=build_order_intent_store(storage))
    engine = TradingEngine(
        exchange=exchange,
        data_provider=None,
        model=None,
        strategy=None,
        risk_manager=None,
        portfolio=portfolio,
        execution=execution,
        exit_manager=exit_manager,
    )
    journal = EventJournal(
        SupabaseEventRepository(connection, table=storage.events_table),
        session_id=storage.journal_session_id,
        source="paper_tp_smoke",
        read_model_repository=SupabaseTradingReadModelRepository(connection),
    )
    await journal.initialize()
    engine.set_event_journal(journal)

    order = Order(
        symbol=position.symbol,
        side=_exit_side(position),
        amount=float(position.amount),
        price=float(exit_price),
        meta={
            ORDER_META_FILL_PRICE_FINAL: True,
            ENGINE_META_EXIT_REASON: reason,
            ENGINE_META_REASON_SUFFIX: "SMOKE-TP",
        },
    )
    command = PlaceOrderCommand(order, reason=reason, ts=tick.ts)
    events = await engine.execute_commands([command])
    print(f"Close events written: {len(events)}")
    for event in events:
        print(type(event).__name__, event)


if __name__ == "__main__":
    asyncio.run(main())
