from dataclasses import dataclass
import pandas as pd

from core.config.base import BaseConfig


@dataclass
class DatasetConfig(BaseConfig):
    exchange: str
    start_ts_ms: int
    end_ts_ms: int
    include_open_interest: bool
    include_funding: bool
    include_premium_index: bool
    output_dir: str

    @classmethod
    def from_env(cls) -> "DatasetConfig":
        def _to_utc_ms(value: str) -> int:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(ts.timestamp() * 1000)

        start_date = cls.env_str("START_DATE", "")
        end_date = cls.env_str("END_DATE", "")
        if start_date and end_date:
            start_ts_ms = _to_utc_ms(start_date)
            end_ts_ms = _to_utc_ms(end_date)
        else:
            start_ts_ms = int(cls.env_str("START_TS_MS", "1704067200000"))
            end_ts_ms = int(cls.env_str("END_TS_MS", "1706745600000"))
        return cls(
            symbols=cls.env_list("SYMBOLS", "BTC/USDT"),
            model_path=cls.env_str("MODEL_PATH", "models/latest.pkl"),
            timeframe=cls.env_str("TIMEFRAME", "1h"),
            htf_timeframe=cls.env_str("HTF_TIMEFRAME", "4h"),
            exchange=cls.env_str("EXCHANGE", "bybit"),
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            include_open_interest=cls.env_str("INCLUDE_OPEN_INTEREST", "1") == "1",
            include_funding=cls.env_str("INCLUDE_FUNDING", "1") == "1",
            include_premium_index=cls.env_str("INCLUDE_PREMIUM_INDEX", "0") == "1",
            output_dir=cls.env_str("DATA_OUTPUT_DIR", "data"),
        )
