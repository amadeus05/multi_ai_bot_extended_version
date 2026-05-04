import pandas as pd

from core.types.domain_types import Tick


def tick_from_kline_event(symbol: str, event) -> Tick:
    close_price = float(event.close_price)
    return Tick(
        symbol=symbol,
        ts=pd.to_datetime(event.end_ms, unit="ms", utc=True).tz_convert(None),
        bid=close_price,
        ask=close_price,
        price=close_price,
        volume=float(event.volume),
        open=float(event.open_price),
        high=float(event.high_price),
        low=float(event.low_price),
        close=close_price,
    )
