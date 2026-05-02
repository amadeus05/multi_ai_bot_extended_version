from __future__ import annotations

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_runtime_factory import build_trading_engine
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from infrastructure.data_providers.websocket_provider import WebSocketProvider


def build_realtime_orchestrator(
    settings,
    exchange: Exchange,
    *,
    model: Model,
) -> RealtimeOrchestrator:
    trading = settings.trading
    data_provider = WebSocketProvider(
        settings.ws_url,
        timeframe=trading.timeframe,
        htf_timeframe=trading.htf_timeframe,
    )
    engine = build_trading_engine(
        settings=trading,
        exchange=exchange,
        data_provider=data_provider,
        model=model,
    )
    return RealtimeOrchestrator(
        engine=engine,
        data_provider=data_provider,
        model=model,
        symbols=list(trading.symbols),
        warmup_bars=max(200, model.required_bars()),
    )
