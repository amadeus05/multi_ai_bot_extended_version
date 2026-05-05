from __future__ import annotations

from dataclasses import dataclass, field

from core.types.domain_types import Order


class BybitExecutionSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class BybitExecutionSafety:
    dry_run: bool = False
    kill_switch: bool = False
    max_order_notional: float | None = None
    allowlist_symbols: frozenset[str] = field(default_factory=frozenset)
    require_private_sync: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        dry_run: bool = False,
        kill_switch: bool = False,
        max_order_notional: float | None = None,
        allowlist_symbols: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
        require_private_sync: bool = False,
    ) -> "BybitExecutionSafety":
        return cls(
            dry_run=bool(dry_run),
            kill_switch=bool(kill_switch),
            max_order_notional=float(max_order_notional) if max_order_notional is not None else None,
            allowlist_symbols=frozenset(_normalize_symbol(symbol) for symbol in (allowlist_symbols or ())),
            require_private_sync=bool(require_private_sync),
        )

    def validate(self, order: Order, *, private_synced: bool = False) -> None:
        symbol = _normalize_symbol(order.symbol)
        if self.kill_switch:
            raise BybitExecutionSafetyError("Bybit execution kill switch is active")
        if self.require_private_sync and not private_synced:
            raise BybitExecutionSafetyError("Bybit private execution stream is not synced")
        if self.allowlist_symbols and symbol not in self.allowlist_symbols:
            raise BybitExecutionSafetyError(f"{symbol} is not in Bybit execution allowlist")
        if self.max_order_notional is not None:
            price = order.price
            if price is None:
                raise BybitExecutionSafetyError("max_order_notional requires an order price before submit")
            notional = abs(float(order.amount) * float(price))
            if notional > float(self.max_order_notional):
                raise BybitExecutionSafetyError(
                    f"order notional {notional:.8f} exceeds max_order_notional {float(self.max_order_notional):.8f}"
                )


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).replace("-", "/").upper()
