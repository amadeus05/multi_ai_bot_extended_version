from __future__ import annotations

from collections.abc import Callable

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_runtime_factory import build_trading_engine
from application.trading_state_restorer import TradingStateRestorer
from core.config.loader import load_storage_settings
from core.config.storage_config import StorageSettings
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.data_providers.websocket_provider import WebSocketProvider
from infrastructure.notifications import build_notifier
from infrastructure.storage.storage_factory import build_event_journal


def build_realtime_orchestrator(
    settings,
    exchange: Exchange,
    *,
    model: Model,
    portfolio: PortfolioManager | None = None,
    execution: ExecutionService | None = None,
    storage: StorageSettings | None = None,
    state_restorer_factory: Callable[..., TradingStateRestorer] | None = None,
) -> RealtimeOrchestrator:
    trading = settings.trading
    storage = storage or load_storage_settings()
    portfolio = portfolio or PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    data_provider = WebSocketProvider(
        settings.ws_url,
        timeframe=trading.timeframe,
        htf_timeframe=trading.htf_timeframe,
    )
    notifier = build_notifier(trading.notifications)
    engine = build_trading_engine(
        settings=trading,
        exchange=exchange,
        data_provider=data_provider,
        model=model,
        portfolio=portfolio,
        execution=execution,
        notifier=notifier,
    )
    journal = build_event_journal(storage)
    warmup_bars = max(200, model.required_bars())
    state_restorer = (
        state_restorer_factory(
            engine=engine,
            portfolio=portfolio,
            data_provider=data_provider,
            symbols=list(trading.symbols),
            required_bars=warmup_bars,
        )
        if state_restorer_factory is not None
        else None
    )
    return RealtimeOrchestrator(
        engine=engine,
        data_provider=data_provider,
        model=model,
        symbols=list(trading.symbols),
        warmup_bars=warmup_bars,
        journal=journal,
        notifier=notifier,
        state_restorer=state_restorer,
    )
