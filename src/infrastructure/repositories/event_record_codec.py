from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pandas as pd

from core.types.event_record import EventRecord


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=_json_default, ensure_ascii=False, sort_keys=True)


def payload_from_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw) if raw else {}
    return value if isinstance(value, dict) else {"value": value}


def record_to_row(record: EventRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "session_id": record.session_id,
        "sequence": int(record.sequence),
        "ts": pd.Timestamp(record.ts).isoformat(),
        "kind": record.kind,
        "source": record.source,
        "payload_json": payload_to_json(record.payload),
    }


def row_to_record(row) -> EventRecord:
    return EventRecord(
        event_id=str(row["event_id"]),
        session_id=str(row["session_id"]),
        sequence=int(row["sequence"]),
        ts=pd.Timestamp(row["ts"]),
        kind=str(row["kind"]),
        source=str(row["source"]),
        payload=payload_from_json(row["payload_json"]),
    )
