import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.training import TrainPipeline
from application.training.artifacts import (
    build_current_run_summary_lines,
    build_recent_runs_table_lines,
    load_train_history,
)
from core.config.train_config import TrainConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = TrainConfig.from_env()
    result = TrainPipeline(cfg).run()
    print("Train finished")
    print(f"median_best_iteration={result.median_best_iteration}")
    print(f"feature_count={len(result.feature_columns)}")
    print(json.dumps(result.oos_metrics, indent=2))
    print("Artifacts:")
    for key, path in result.artifact_paths.items():
        print(f"- {key}: {path}")
    history_path = result.artifact_paths.get("train_history")
    if history_path:
        history = load_train_history(Path(history_path))
        if history:
            print("=" * 72)
            print("Current training summary:")
            for line in build_current_run_summary_lines(history[-1]):
                print(line)
            print(f"Recent training runs (latest {min(10, len(history))}):")
            for line in build_recent_runs_table_lines(history, limit=10):
                print(line)


if __name__ == "__main__":
    main()
