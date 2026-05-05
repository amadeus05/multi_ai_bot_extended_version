from __future__ import annotations

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_runtime_factory import build_trading_engine
from core.config.loader import load_storage_settings
from core.interfaces.model import Model
from core.interfaces.exchange import Exchange
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
) -> RealtimeOrchestrator:
    trading = settings.trading
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
        notifier=notifier,
    )
    journal = build_event_journal(load_storage_settings())
    return RealtimeOrchestrator(
        engine=engine,
        data_provider=data_provider,
        model=model,
        symbols=list(trading.symbols),
        warmup_bars=max(200, model.required_bars()),
        journal=journal,
        notifier=notifier,
    )
