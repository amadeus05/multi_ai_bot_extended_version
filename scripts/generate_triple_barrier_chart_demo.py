from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from triple_barrier_chart import render_triple_barrier_chart  # noqa: E402


def build_demo_candles(count: int = 200) -> list[dict[str, float | str]]:
    candles: list[dict[str, float | str]] = []
    start_time = datetime(2025, 4, 2, 19, 0)
    price = 1900.0

    for i in range(count):
        if i < 35:
            target = 1900.0 - i * 3.1 + math.sin(i / 3) * 11
        elif i < 75:
            target = 1790.0 + (i - 35) * 1.2 + math.sin(i / 4) * 8
        elif i < 125:
            target = 1835.0 - (i - 75) * 0.7 + math.sin(i / 5) * 7
        elif i < 150:
            target = 1802.0 + math.sin(i / 4) * 4
        elif i < 170:
            target = 1810.0 - (i - 150) * 2.1 + math.sin(i / 2) * 5
        else:
            target = 1768.0 + math.sin(i / 5) * 11 + (i - 170) * 0.25

        open_price = price
        close = target
        high = max(open_price, close) + 5.5
        low = min(open_price, close) - 5.5

        if i == 150:
            open_price = 1816.0
            close = 1810.0
            high = 1820.0
            low = 1806.0
        elif i == 169:
            open_price = 1778.0
            close = 1762.0
            high = 1782.0
            low = 1754.0

        candles.append(
            {
                "time": (start_time + timedelta(minutes=15 * i)).strftime("%d.%m %H:%M"),
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
            }
        )
        price = close

    return candles


if __name__ == "__main__":
    candles = build_demo_candles(200)
    entry_index = 150
    entry = 1810.0
    stop = 1832.73
    take = 1764.27

    entry_result = render_triple_barrier_chart(
        candles[: entry_index + 1],
        entry=entry,
        stop=stop,
        take=take,
        probability=0.75,
        output_path=ROOT_DIR / "charts" / "test_triple_barrier_entry.png",
        title="ETHUSDT",
    )
    hit_result = render_triple_barrier_chart(
        candles,
        entry=entry,
        stop=stop,
        take=take,
        entry_index=entry_index,
        probability=0.75,
        output_path=ROOT_DIR / "charts" / "test_triple_barrier_hit.png",
        title="ETHUSDT",
    )
    legacy_result = render_triple_barrier_chart(
        candles,
        entry=1810.0,
        stop=1832.73,
        take=1764.27,
        entry_index=entry_index,
        probability=0.75,
        output_path=ROOT_DIR / "charts" / "test_triple_barrier.png",
        title="ETHUSDT",
    )
    print(f"entry chart saved: {entry_result.output_path}")
    print(f"hit chart saved: {hit_result.output_path}")
    print(f"legacy chart saved: {legacy_result.output_path}")
    print(f"hit break: {hit_result.break_event}")
