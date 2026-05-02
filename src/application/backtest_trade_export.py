from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPORT_COLUMNS = [
    "source",
    "seq",
    "trade_number",
    "symbol",
    "direction",
    "reason",
    "open_ts",
    "close_ts",
    "entry_price",
    "exit_price",
    "qty",
    "pnl_pct",
    "pnl_abs",
    "commission",
]


def _direction_from_row(row: dict) -> str:
    direction = row.get("direction")
    if direction:
        return str(direction).upper()
    side = str(row.get("side", "")).lower()
    if side == "long":
        return "LONG"
    if side == "short":
        return "SHORT"
    return ""


def _open_event_lookup(report: dict) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for event in report.get("trade_events", []) or []:
        if event.get("type") != "OPEN":
            continue
        try:
            trade_number = int(event.get("trade_number"))
        except (TypeError, ValueError):
            continue
        lookup[trade_number] = event
    return lookup


def export_closed_trades_csv(report: dict, output_path: str | Path, *, source: str) -> Path:
    opens = _open_event_lookup(report)
    rows = []
    for seq, row in enumerate(report.get("closed_trades", []) or [], start=1):
        try:
            trade_number = int(row.get("trade_number", seq))
        except (TypeError, ValueError):
            trade_number = seq
        open_event = opens.get(trade_number, {})
        rows.append(
            {
                "source": source,
                "seq": seq,
                "trade_number": trade_number,
                "symbol": row.get("symbol", ""),
                "direction": _direction_from_row(row),
                "reason": row.get("reason", ""),
                "open_ts": open_event.get("ts", row.get("open_ts", "")),
                "close_ts": row.get("ts", row.get("close_ts", "")),
                "entry_price": row.get("entry_price", open_event.get("price", "")),
                "exit_price": row.get("exit_price", ""),
                "qty": row.get("qty", open_event.get("qty", "")),
                "pnl_pct": row.get("pnl_pct", ""),
                "pnl_abs": row.get("pnl_abs", ""),
                "commission": row.get("commission", ""),
            }
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=EXPORT_COLUMNS).to_csv(out, index=False)
    print(f"Saved closed trades CSV: {out.resolve()}")
    return out
