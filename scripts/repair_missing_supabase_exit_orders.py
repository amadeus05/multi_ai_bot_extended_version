from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any

import requests


def load_env(path: str = ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


class SupabaseRest:
    def __init__(self, *, url: str, service_key: str, schema: str) -> None:
        if not url or not service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._base = url.rstrip("/") + "/rest/v1"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }

    def get_all(self, table: str, select: str, extra: dict[str, str] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            params = {"select": select, "offset": str(offset), "limit": str(page_size)}
            if extra:
                params.update(extra)
            response = requests.get(f"{self._base}/{table}", headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise RuntimeError(f"unexpected payload from {table}")
            rows.extend(batch)
            if len(batch) < page_size:
                return rows
            offset += page_size

    def insert_ignore(self, table: str, row: dict[str, Any], *, on_conflict: str) -> bool:
        headers = dict(self._headers)
        headers["Prefer"] = "resolution=ignore-duplicates,return=representation"
        response = requests.post(
            f"{self._base}/{table}",
            headers=headers,
            params={"on_conflict": on_conflict},
            json=row,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json() if response.content else []
        return bool(payload)


def build_repair_rows(rest: SupabaseRest, session_id: str | None) -> list[dict[str, Any]]:
    session_filter = {"session_id": f"eq.{session_id}"} if session_id else None
    orders = rest.get_all("orders", "session_id,order_id", session_filter)
    fills = rest.get_all(
        "fills",
        "session_id,sequence,fill_id,order_id,client_order_id,symbol,side,amount,price,ts,command_reason,meta_json",
        session_filter,
    )
    closed = rest.get_all(
        "closed_trades",
        "id,session_id,trade_id,symbol,exit_order_id,exit_fill_id,exit_reason",
        session_filter,
    )

    order_keys = {(row["session_id"], row["order_id"]) for row in orders if row.get("order_id")}
    fills_by_key = {(row["session_id"], row["fill_id"]): row for row in fills if row.get("fill_id")}

    repair_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for trade in closed:
        key = (trade.get("session_id"), trade.get("exit_order_id"))
        if not key[0] or not key[1] or key in order_keys or key in seen:
            continue
        fill = fills_by_key.get((trade["session_id"], trade.get("exit_fill_id")))
        if fill is None:
            continue
        if fill.get("order_id") != trade.get("exit_order_id"):
            continue

        meta_json = fill.get("meta_json") if isinstance(fill.get("meta_json"), dict) else {}
        repair_rows.append(
            {
                "session_id": fill["session_id"],
                "sequence": int(fill["sequence"]),
                "order_id": fill["order_id"],
                "client_order_id": fill.get("client_order_id"),
                "symbol": fill["symbol"],
                "side": fill["side"],
                "order_type": "market",
                "status": "filled",
                "reason": fill.get("command_reason") or trade.get("exit_reason"),
                "amount": float(fill["amount"]),
                "price": float(fill["price"]),
                "created_ts": fill["ts"],
                "updated_ts": fill["ts"],
                "source": "repair_missing_exit_order",
                "meta_json": meta_json,
            }
        )
        seen.add(key)
    return repair_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair missing Supabase orders for closed trade exit fills.")
    parser.add_argument("--apply", action="store_true", help="Write repair rows. Without this flag, only prints the plan.")
    parser.add_argument("--session-id", default="", help="Optional session id. Defaults to EVENT_JOURNAL_SESSION_ID.")
    args = parser.parse_args()

    env = {**load_env(), **os.environ}
    session_id = args.session_id or env.get("EVENT_JOURNAL_SESSION_ID") or ""
    rest = SupabaseRest(
        url=env.get("SUPABASE_URL", ""),
        service_key=env.get("SUPABASE_SERVICE_KEY", ""),
        schema=env.get("SUPABASE_SCHEMA", "public"),
    )

    repair_rows = build_repair_rows(rest, session_id or None)
    print(f"session={session_id or '<all>'}")
    print(f"missing_exit_orders={len(repair_rows)}")
    for row in repair_rows:
        print(
            "plan order_id={order_id} client_order_id={client_order_id} symbol={symbol} "
            "side={side} reason={reason} amount={amount} price={price} ts={created_ts}".format(**row)
        )

    if not args.apply:
        print("dry_run=true")
        return 0

    inserted = 0
    skipped = 0
    for row in repair_rows:
        if rest.insert_ignore("orders", row, on_conflict="session_id,order_id"):
            inserted += 1
        else:
            skipped += 1
    print(f"inserted={inserted} skipped_existing={skipped}")
    print("by_symbol=" + str(dict(Counter(row["symbol"] for row in repair_rows))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
