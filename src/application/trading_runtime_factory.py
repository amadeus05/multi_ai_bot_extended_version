from __future__ import annotations

from dataclasses import replace

from core.config.settings import TradingSettings
from core.interfaces.data_provider import DataProvider
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from core.interfaces.notifier import Notifier
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.ml.labeling.barrier_policy import BarrierPolicy
from domain.ml.labeling.models import LabelingConfig
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.adaptive_risk_manager import AdaptiveRiskManager
from domain.risk.models.risk_profile import RiskProfile
from domain.strategy import BacktestParitySignalStrategy
from infrastructure.notifications import build_notifier
from application.trading_engine import TradingEngine


def build_labeling_config(settings: TradingSettings) -> LabelingConfig:
    cfg = LabelingConfig.from_env()
    return replace(
        cfg,
        barrier=replace(
            cfg.barrier,
            taker_com=float(settings.costs.taker_com),
            slippage=float(settings.costs.slippage),
        ),
    )


def build_strategy(settings: TradingSettings) -> BacktestParitySignalStrategy:
    return BacktestParitySignalStrategy(
        directional_proba_threshold=float(settings.signal.directional_proba_threshold),
        min_signal_gap=float(settings.signal.min_signal_gap),
        allow_longs=bool(settings.signal.allow_longs),
        allow_shorts=bool(settings.signal.allow_shorts),
    )


def build_trading_engine(
    *,
    settings: TradingSettings,
    exchange: Exchange,
    data_provider: DataProvider,
    model: Model,
    portfolio: PortfolioManager | None = None,
    execution: ExecutionService | None = None,
    notifier: Notifier | None = None,
) -> TradingEngine:
    labeling_cfg = build_labeling_config(settings)
    barrier_policy = BarrierPolicy(labeling_cfg)
    portfolio = portfolio or PortfolioManager(cash={"USDT": float(settings.initial_capital)})
    execution = execution or ExecutionService()
    notifier = notifier or build_notifier(settings.notifications)
    return TradingEngine(
        exchange=exchange,
        data_provider=data_provider,
        model=model,
        strategy=build_strategy(settings),
        risk_manager=AdaptiveRiskManager(
            profile=RiskProfile.from_settings(settings.risk, leverage=settings.leverage)
        ),
        portfolio=portfolio,
        execution=execution,
        exit_manager=ExitManager(slippage=barrier_policy.slippage),
        barrier_policy=barrier_policy,
        notifier=notifier,
    )
