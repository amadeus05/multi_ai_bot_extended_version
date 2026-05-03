from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pandas as pd

from core.types.commands import CancelOrderCommand, PlaceOrderCommand, TradingCommand
from core.types.events import TradingEvent


def event_kind(event: TradingEvent) -> str:
    return type(event).__name__


def command_kind(command: TradingCommand) -> str:
    return type(command).__name__


def event_record_id(event: TradingEvent) -> str | None:
    value = getattr(event, "event_id", None)
    return str(value) if value else None


def command_record_id(command: TradingCommand) -> str:
    if isinstance(command, PlaceOrderCommand):
        order = command.order
        return f"command:place:{order.client_order_id or id(command)}"
    if isinstance(command, CancelOrderCommand):
        return f"command:cancel:{command.order_id}:{command.reason or ''}"
    return f"command:{type(command).__name__}:{id(command)}"


def serialize_event(event: TradingEvent) -> dict[str, Any]:
    return _normalize(event)


def serialize_command(command: TradingCommand) -> dict[str, Any]:
    return _normalize(command)


def object_ts(value) -> pd.Timestamp:
    ts = getattr(value, "ts", None)
    return pd.Timestamp(ts) if ts is not None else pd.Timestamp.utcnow()


def _normalize(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
