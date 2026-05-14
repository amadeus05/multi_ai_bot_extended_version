from __future__ import annotations

from collections.abc import Callable

from application.modeling.adapters.lightgbm import LightGbmWalkForwardAdapter
from application.modeling.contracts import WalkForwardAdapter
from core.config.train_config import TrainConfig

AdapterFactory = Callable[[TrainConfig], WalkForwardAdapter]


WALK_FORWARD_ADAPTERS: dict[str, AdapterFactory] = {
    LightGbmWalkForwardAdapter.model_id: lambda train_cfg: LightGbmWalkForwardAdapter(train_cfg=train_cfg),
}


def create_walk_forward_adapter(model_id: str, *, train_cfg: TrainConfig) -> WalkForwardAdapter:
    key = model_id.strip().lower()
    try:
        factory = WALK_FORWARD_ADAPTERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(WALK_FORWARD_ADAPTERS))
        raise ValueError(f"Unknown walk-forward model adapter {model_id!r}. Available: {available}") from exc
    return factory(train_cfg)
