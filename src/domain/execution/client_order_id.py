from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import pandas as pd

from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order


MAX_CLIENT_ORDER_ID_LEN = 36


def ensure_client_order_id(order: Order, command: PlaceOrderCommand | None = None) -> None:
    if order.client_order_id:
        return
    if command is not None:
        deterministic = deterministic_client_order_id(command)
        if deterministic is not None:
            order.client_order_id = deterministic
            return
    order.client_order_id = str(uuid.uuid4())


def deterministic_client_order_id(command: PlaceOrderCommand) -> str | None:
    order = command.order
    intent_ts = _intent_timestamp(command)
    if intent_ts is None:
        return None

    symbol = _api_symbol(order.symbol)
    side = order.side.value
    order_type = order.type.value
    reason = _clean_part(command.reason or "order", max_len=8)
    fingerprint = _fingerprint(command)
    prefix = _clean_part(f"{reason}-{symbol}-{side[0]}-{order_type[0]}", max_len=18)
    stamp = pd.to_datetime(intent_ts).strftime("%y%m%d%H%M")
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8]
    client_order_id = f"{prefix}-{stamp}-{digest}"
    return client_order_id[:MAX_CLIENT_ORDER_ID_LEN]


def _intent_timestamp(command: PlaceOrderCommand):
    if command.source_tick is not None:
        return command.source_tick.ts
    return command.ts


def _fingerprint(command: PlaceOrderCommand) -> str:
    order = command.order
    meta = _stable_meta(order.meta or {})
    payload = {
        "symbol": _api_symbol(order.symbol),
        "side": order.side.value,
        "amount": _float_string(order.amount),
        "price": _float_string(order.price) if order.price is not None else None,
        "type": order.type.value,
        "reason": command.reason or "",
        "ts": str(pd.to_datetime(_intent_timestamp(command))),
        "meta": meta,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _stable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    stable_keys = (
        "signal_number",
        "p_long",
        "p_short",
        "score",
        "signal_gap",
        "direction_prob",
        "directional_proba_threshold",
        "min_signal_gap",
        "barrier_stop_pct",
        "barrier_take_pct",
    )
    return {key: meta[key] for key in stable_keys if key in meta}


def _api_symbol(symbol: str) -> str:
    return str(symbol).replace("-", "/").replace("/", "").upper()


def _clean_part(value: str, *, max_len: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return cleaned[:max_len] or "order"


def _float_string(value: float) -> str:
    return format(float(value), ".12g")
