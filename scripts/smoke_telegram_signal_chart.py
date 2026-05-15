from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.trading_engine import SIGNAL_CHART_PATH  # noqa: E402
from core.config.loader import load_paper_settings  # noqa: E402
from core.types.notifications import SignalNotification  # noqa: E402
from infrastructure.notifications.notifiers import TelegramNotifier  # noqa: E402
from triple_barrier_chart import render_triple_barrier_chart  # noqa: E402


def build_demo_candles(count: int = 220) -> list[dict[str, float | str]]:
    candles: list[dict[str, float | str]] = []
    start_time = datetime.now(UTC) - timedelta(minutes=15 * count)
    price = 100.0

    for i in range(count):
        drift = i * 0.035
        wave = math.sin(i / 7) * 1.25
        close = 100.0 + drift + wave
        open_price = price
        high = max(open_price, close) + 0.55
        low = min(open_price, close) - 0.55
        candles.append(
            {
                "timestamp": (start_time + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
            }
        )
        price = close

    return candles


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally send a test Telegram signal chart.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol title for the test notification.")
    parser.add_argument("--side", choices=("LONG", "SHORT"), default="LONG", help="Signal side.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate the chart, do not send Telegram photo.")
    args = parser.parse_args()

    candles = build_demo_candles()
    entry = float(candles[-1]["close"])
    if args.side == "LONG":
        stop = entry * 0.985
        take = entry * 1.03
        p_long, p_short = 0.78, 0.22
    else:
        stop = entry * 1.015
        take = entry * 0.97
        p_long, p_short = 0.22, 0.78

    chart_path = PROJECT_ROOT / SIGNAL_CHART_PATH
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    render_triple_barrier_chart(
        candles,
        entry=entry,
        stop=stop,
        take=take,
        probability=max(p_long, p_short),
        output_path=chart_path,
        title=args.symbol,
    )

    notification = SignalNotification(
        signal_id=999,
        symbol=args.symbol,
        side=args.side,
        ts=pd.Timestamp.now(tz="UTC"),
        entry_price=entry,
        amount=0.01,
        stop_price=stop,
        take_price=take,
        stop_pct=abs(entry - stop) / entry,
        take_pct=abs(take - entry) / entry,
        p_long=p_long,
        p_short=p_short,
        signal_gap=abs(p_long - p_short),
        direction_prob=max(p_long, p_short),
        proba_threshold=0.55,
        min_signal_gap=0.05,
        balance=100.0,
        chart_path=chart_path,
    )

    print(f"Chart saved: {chart_path.resolve()}")
    if args.dry_run:
        print("Dry run: Telegram send skipped.")
        return

    settings = load_paper_settings()
    telegram = settings.trading.notifications
    if not telegram.telegram_bot_token or not telegram.telegram_chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env before sending.")

    notifier = TelegramNotifier(
        bot_token=telegram.telegram_bot_token,
        chat_id=telegram.telegram_chat_id,
        fail_silently=False,
    )
    await notifier.notify_signal(notification)
    print("Telegram test signal with chart sent.")


if __name__ == "__main__":
    asyncio.run(main())
