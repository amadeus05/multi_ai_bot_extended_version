import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.training import CandleLSTMConfig, CandleLSTMTrainPipeline
from core.config.train_config import TrainConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    train_cfg = TrainConfig.from_env()
    lstm_cfg = CandleLSTMConfig.from_env()
    result = CandleLSTMTrainPipeline(train_cfg=train_cfg, lstm_cfg=lstm_cfg).run()
    print("Candle LSTM train finished")
    print(f"feature_count={len(result.feature_columns)}")
    print(json.dumps(result.oos_metrics, indent=2))
    print("Artifacts:")
    for key, path in result.artifact_paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
