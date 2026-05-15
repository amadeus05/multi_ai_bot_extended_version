import pandas as pd

from core.types.notifications import SignalNotification
from infrastructure.notifications.formatters import format_signal_text


def test_signal_prices_use_more_precision_for_low_price_symbols() -> None:
    text = format_signal_text(
        SignalNotification(
            signal_id=2,
            symbol="DOGE/USDT",
            side="LONG",
            ts=pd.Timestamp("2026-05-14T15:59:00"),
            entry_price=0.11000,
            amount=149.874913,
            stop_price=0.10681,
            take_price=0.11627,
            stop_pct=0.029,
            take_pct=0.057,
            p_long=0.575,
            p_short=0.425,
            signal_gap=0.149,
            direction_prob=0.575,
            proba_threshold=0.55,
            min_signal_gap=0.05,
            balance=97.92,
        )
    )

    assert "Цена     0.11000" in text
    assert "🛑 Стоп   0.10681" in text
    assert "🎯 Тейк   0.11627" in text


def test_signal_prices_stay_compact_for_high_price_symbols() -> None:
    text = format_signal_text(
        SignalNotification(
            signal_id=1,
            symbol="BTC/USDT",
            side="LONG",
            ts=pd.Timestamp("2026-05-14T15:59:00"),
            entry_price=102345.678,
            amount=0.001,
            stop_price=99377.653,
            take_price=108179.38,
            stop_pct=0.029,
            take_pct=0.057,
            p_long=0.575,
            p_short=0.425,
            signal_gap=0.149,
            direction_prob=0.575,
            proba_threshold=0.55,
            min_signal_gap=0.05,
            balance=97.92,
        )
    )

    assert "Цена     102 345.68" in text
    assert "🛑 Стоп   99 377.65" in text
    assert "🎯 Тейк   108 179.38" in text
