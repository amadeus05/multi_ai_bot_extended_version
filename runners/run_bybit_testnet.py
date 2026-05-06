import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "runners"))

from application.inference.model_factory import load_inference_model
from application.trading_state_restorer import ExchangeSnapshotStateRestorer
from core.config.loader import load_live_settings, load_storage_settings
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.bybit.bybit_execution_exchange import BybitExecutionExchange
from infrastructure.exchanges.bybit.bybit_execution_safety import BybitExecutionSafety
from infrastructure.storage.storage_factory import build_order_intent_store
from realtime_common import build_realtime_orchestrator


def build_testnet_safety_from_env(symbols: tuple[str, ...] | list[str]) -> BybitExecutionSafety:
    allowlist_raw = os.getenv("BYBIT_TESTNET_ALLOWLIST", ",".join(symbols))
    allowlist = tuple(item.strip() for item in allowlist_raw.split(",") if item.strip())
    max_notional_raw = os.getenv("BYBIT_TESTNET_MAX_ORDER_NOTIONAL", "50")
    return BybitExecutionSafety.from_values(
        dry_run=_env_bool("BYBIT_TESTNET_DRY_RUN", True),
        kill_switch=_env_bool("BYBIT_TESTNET_KILL_SWITCH", False),
        max_order_notional=float(max_notional_raw) if max_notional_raw.strip() else None,
        allowlist_symbols=allowlist,
        require_private_sync=True,
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


async def main() -> None:
    settings = load_live_settings(overrides={"testnet": True})
    trading = settings.trading
    model = load_inference_model(cwd=Path.cwd(), trading=trading)
    exchange = BybitExecutionExchange(
        settings.api_key,
        settings.api_secret,
        testnet=True,
        safety=build_testnet_safety_from_env(trading.symbols),
    )
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    order_intent_store = build_order_intent_store(load_storage_settings())
    execution = ExecutionService(order_intent_store=order_intent_store)
    orchestrator = build_realtime_orchestrator(
        settings=settings,
        exchange=exchange,
        model=model,
        portfolio=portfolio,
        execution=execution,
    )
    orchestrator.set_state_restorer(
        ExchangeSnapshotStateRestorer(
            exchange=exchange,
            portfolio=portfolio,
            order_intent_store=order_intent_store,
        )
    )
    orchestrator.set_execution_event_source(exchange.stream_private_events)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
