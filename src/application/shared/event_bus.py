from collections import defaultdict
from typing import Any, Awaitable, Callable


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        self._handlers[event_name].append(handler)

    async def publish(self, event_name: str, event: Any) -> None:
        for handler in self._handlers.get(event_name, []):
            await handler(event)
