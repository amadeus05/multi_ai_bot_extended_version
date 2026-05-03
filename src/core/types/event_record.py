from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    session_id: str
    sequence: int
    ts: pd.Timestamp
    kind: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
