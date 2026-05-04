from __future__ import annotations

import html

import pandas as pd

from core.types.notifications import SignalNotification, SystemNotification, TradeExitNotification


def format_signal_text(notification: SignalNotification) -> str:
    notional = notification.entry_price * notification.amount
    return "\n".join(
        [
            f"[SIGNAL #{notification.signal_id}] {notification.symbol} {notification.side.upper()}",
            "",
            f"Time: {_fmt_ts(notification.ts)}",
            f"Entry: {_fmt_num(notification.entry_price)}",
            f"Amount: {_fmt_num(notification.amount, 6)}",
            f"Notional: {_fmt_balance(notional)}",
            f"Balance: {_fmt_balance(notification.balance)}",
            "",
            "Risk:",
            f"Stop: {_fmt_num(notification.stop_price)} ({_fmt_pct(_neg(notification.stop_pct))})",
            f"Take: {_fmt_num(notification.take_price)} ({_fmt_pct(notification.take_pct)})",
            "",
            "Model:",
            f"p_long: {_fmt_num(notification.p_long, 3)}",
            f"p_short: {_fmt_num(notification.p_short, 3)}",
            f"direction_prob: {_fmt_num(notification.direction_prob, 3)}",
            f"gap: {_fmt_num(notification.signal_gap, 3)}",
            f"threshold: {_fmt_num(notification.proba_threshold, 3)}",
            f"min_gap: {_fmt_num(notification.min_signal_gap, 3)}",
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
    return _html(format_signal_text(notification)).replace("\n", "<br>")


def format_trade_exit_html(notification: TradeExitNotification) -> str:
    return _html(format_trade_exit_text(notification)).replace("\n", "<br>")


def format_system_html(notification: SystemNotification) -> str:
    return _html(format_system_text(notification)).replace("\n", "<br>")


def _fmt_ts(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "n/a"
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+,.2f} USDT"


def _fmt_balance(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.2f} USDT"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def _html(text: str) -> str:
    return html.escape(text, quote=False)


def _neg(value: float | None) -> float | None:
    if value is None:
        return None
    return -float(value)
