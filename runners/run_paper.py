import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.inference.lightgbm_inference_model import load_default_lightgbm_model
from core.config.paper_config import PaperConfig
from domain.strategy import BacktestParitySignalStrategy
from infrastructure.exchanges.simulation.simulated_exchange import SimulatedExchange
from runners.realtime_common import build_realtime_orchestrator


async def main() -> None:
    cfg = PaperConfig.from_env()
    model = load_default_lightgbm_model(cwd=Path.cwd(), model_path_cfg=cfg.model_path, required_bars=250)
    strategy = BacktestParitySignalStrategy(
        directional_proba_threshold=cfg.directional_proba_threshold,
        min_signal_gap=cfg.min_signal_gap,
        allow_longs=cfg.allow_longs,
        allow_shorts=cfg.allow_shorts,
    )
    exchange = SimulatedExchange()
    orchestrator = build_realtime_orchestrator(
        cfg=cfg,
        exchange=exchange,
        initial_capital=cfg.initial_capital,
        model=model,
        strategy=strategy,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
