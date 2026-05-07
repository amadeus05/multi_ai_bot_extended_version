import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "runners"))

from application.inference.lightgbm_inference_model import load_default_lightgbm_model
from application.paper_state_restorer import PaperStateRestorer
from core.config.loader import load_paper_settings, load_storage_settings
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from infrastructure.storage.storage_factory import build_order_intent_store
from realtime_common import build_realtime_orchestrator


async def main() -> None:
    settings = load_paper_settings()
    storage = load_storage_settings()
    trading = settings.trading
    model = load_default_lightgbm_model(cwd=Path.cwd(), model_path_cfg=trading.model_path, required_bars=250)
    exchange = SimulatedExchange(
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    order_intent_store = build_order_intent_store(storage)
    execution = ExecutionService(order_intent_store=order_intent_store)
    orchestrator = build_realtime_orchestrator(
        settings=settings,
        exchange=exchange,
        model=model,
        portfolio=portfolio,
        execution=execution,
        storage=storage,
        state_restorer_factory=lambda **deps: PaperStateRestorer(
            storage=storage,
            exit_manager=ExitManager(slippage=float(trading.costs.slippage)),
            **deps,
        ),
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
