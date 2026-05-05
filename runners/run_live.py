import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "runners"))

from application.inference.lightgbm_inference_model import load_default_lightgbm_model
from application.trading_state_restorer import ExchangeSnapshotStateRestorer
from core.config.loader import load_live_settings
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.bybit.bybit_execution_exchange import BybitExecutionExchange
from realtime_common import build_realtime_orchestrator


async def main() -> None:
    settings = load_live_settings()
    trading = settings.trading
    model = load_default_lightgbm_model(cwd=Path.cwd(), model_path_cfg=trading.model_path, required_bars=250)
    exchange = BybitExecutionExchange(settings.api_key, settings.api_secret, testnet=settings.testnet)
    portfolio = PortfolioManager(cash={"USDT": float(trading.initial_capital)})
    orchestrator = build_realtime_orchestrator(
        settings=settings,
        exchange=exchange,
        model=model,
        portfolio=portfolio,
    )
    orchestrator.set_state_restorer(
        ExchangeSnapshotStateRestorer(exchange=exchange, portfolio=portfolio)
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
