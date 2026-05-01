from pathlib import Path

import pandas as pd


class ParquetWriter:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    def write(self, frame: pd.DataFrame, relative_path: str) -> Path:
        target_path = self.base_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target_path, index=False)
        return target_path
