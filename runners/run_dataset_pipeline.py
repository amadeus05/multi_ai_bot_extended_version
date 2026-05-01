import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.pipelines.dataset_pipeline import DatasetPipeline
from core.config.dataset_config import DatasetConfig
from domain.ml.labeling import LabelingConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = DatasetConfig.from_env()
    labeling_cfg = LabelingConfig.from_env()
    pipeline = DatasetPipeline(cfg, labeling_cfg=labeling_cfg)
    outputs = pipeline.run()
    print(f"Built {len(outputs)} labeled dataset files")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
