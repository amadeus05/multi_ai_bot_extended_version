from __future__ import annotations

import html

import pandas as pd

from core.types.notifications import SignalNotification, SystemNotification, TradeExitNotification


def format_signal_text(notification: SignalNotification) -> str:
    notional = notification.entry_price * notification.amount
    base_asset = notification.symbol.split("/")[0]
    return "\n".join(
        [
            f"🔬 СИГНАЛ #{notification.signal_id} • {notification.symbol}",
            f"▲ {notification.side.upper()}  {_fmt_ts_short(notification.ts)}",
            "",
            "Вход",
            f"Цена     {_fmt_plain_num(notification.entry_price, 2)}",
            f"Объём    {_fmt_num(notification.amount, 6)} {base_asset}  ({_fmt_balance_short(notional)})",
            f"Баланс   {_fmt_balance(notification.balance)}",
            "",
            f"🛑 Стоп   {_fmt_plain_num(notification.stop_price, 2)}  - {_fmt_pct_abs(notification.stop_pct)}",
            f"🎯 Тейк   {_fmt_plain_num(notification.take_price, 2)}  + {_fmt_pct_abs(notification.take_pct)}",
            "",
            f"🤖 Long {_fmt_pct_abs(notification.p_long)}  Short {_fmt_pct_abs(notification.p_short)}  gap {_fmt_pct_abs(notification.signal_gap)}",
        ]
    )


def format_trade_exit_text(notification: TradeExitNotification) -> str:
    return "\n".join(
        [
            f"[TRADE CLOSED #{notification.trade_number}] {notification.symbol} {notification.side.upper()}",
            "",
            f"Reason: {notification.reason}",
            f"Time: {_fmt_ts(notification.ts)}",
            f"Entry: {_fmt_num(notification.entry_price)}",
            f"Exit: {_fmt_num(notification.exit_price)}",
            f"Qty: {_fmt_num(notification.qty, 6)}",
            "",
            f"PnL: {_fmt_money(notification.pnl_abs)}",
            f"PnL %: {_fmt_pct(notification.pnl_pct)}",
            f"Commission: {_fmt_balance(notification.commission)}",
            f"Balance: {_fmt_balance(notification.balance)}",
        ]
    )


def format_system_text(notification: SystemNotification) -> str:
    lines = [
        f"[{notification.level.upper()}] {notification.title}",
        "",
        notification.message,
    ]
    if notification.where:
        lines.append(f"Where: {notification.where}")
    if notification.error:
        lines.append(f"Error: {notification.error}")
    return "\n".join(lines)


def format_signal_html(notification: SignalNotification) -> str:
    return _html(format_signal_text(notification))


def format_trade_exit_html(notification: TradeExitNotification) -> str:
    return _html(format_trade_exit_text(notification))


def format_system_html(notification: SystemNotification) -> str:
    return _html(format_system_text(notification))


def _fmt_ts(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "n/a"
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_ts_short(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "n/a"
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%d.%m.%Y · %H:%M")


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _fmt_plain_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}".replace(",", " ")


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+,.2f} USDT"


def _fmt_balance(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.2f} USDT"


def _fmt_balance_short(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.0f} USDT"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def _fmt_pct_abs(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{abs(float(value)) * 100:.1f}%"


def _html(text: str) -> str:
    return html.escape(text, quote=False)


def _neg(value: float | None) -> float | None:
    if value is None:
        return None
    return -float(value)
