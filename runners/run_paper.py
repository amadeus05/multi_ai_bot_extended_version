import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "runners"))

from application.inference.model_factory import load_inference_model
from core.config.loader import load_paper_settings
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from realtime_common import build_realtime_orchestrator


async def main() -> None:
    settings = load_paper_settings()
    trading = settings.trading
    model = load_inference_model(cwd=Path.cwd(), trading=trading)
    exchange = SimulatedExchange(
        commission=float(trading.costs.taker_com),
        slippage=float(trading.costs.slippage),
        leverage=float(trading.leverage),
    )
    orchestrator = build_realtime_orchestrator(
        settings=settings,
        exchange=exchange,
        model=model,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
