import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.inference.lightgbm_inference_model import load_default_lightgbm_model
from core.config.loader import load_live_settings
from infrastructure.exchanges.bybit.bybit_service import BybitService
from runners.realtime_common import build_realtime_orchestrator


async def main() -> None:
    settings = load_live_settings()
    trading = settings.trading
    model = load_default_lightgbm_model(cwd=Path.cwd(), model_path_cfg=trading.model_path, required_bars=250)
    exchange = BybitService(settings.api_key, settings.api_secret, testnet=settings.testnet)
    orchestrator = build_realtime_orchestrator(
        settings=settings,
        exchange=exchange,
        model=model,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
