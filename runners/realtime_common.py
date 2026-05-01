from __future__ import annotations

from application.realtime_orchestrator import RealtimeOrchestrator
from application.trading_engine import TradingEngine
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.adaptive_risk_manager import AdaptiveRiskManager
from domain.risk.models.risk_profile import RiskProfile
from domain.strategy.strategy_engine import Strategy
from domain.execution.exit_manager import ExitManager
from infrastructure.data_providers.websocket_provider import WebSocketProvider


def build_realtime_orchestrator(
    cfg,
    exchange: Exchange,
    initial_capital: float,
    *,
    model: Model,
    strategy: Strategy,
) -> RealtimeOrchestrator:
    data_provider = WebSocketProvider(cfg.ws_url, timeframe=cfg.timeframe, htf_timeframe=cfg.htf_timeframe)
    risk_manager = AdaptiveRiskManager(
        profile=RiskProfile(
            leverage=cfg.leverage,
            risk_per_trade=cfg.risk_per_trade,
        )
    )
    engine = TradingEngine(
        exchange=exchange,
        data_provider=data_provider,
        model=model,
        strategy=strategy,
        risk_manager=risk_manager,
        portfolio=PortfolioManager(cash={"USDT": initial_capital}),
        execution=ExecutionService(),
        exit_manager=ExitManager(slippage=getattr(cfg, "slippage", 0.0003)),
    )
    return RealtimeOrchestrator(
        engine=engine,
        data_provider=data_provider,
        model=model,
        symbols=cfg.symbols,
        warmup_bars=max(200, model.required_bars()),
    )
