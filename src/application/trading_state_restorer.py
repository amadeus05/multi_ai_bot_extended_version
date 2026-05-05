from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from core.interfaces.exchange import Exchange
from domain.portfolio.portfolio_manager import PortfolioManager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestoreResult:
    restored: bool
    source: str
    positions_count: int = 0
    open_orders_count: int = 0
    recent_executions_count: int = 0
    message: str = ""


class TradingStateRestorer(Protocol):
    async def restore_trading_state(self) -> RestoreResult:
        ...


class NoopTradingStateRestorer:
    def __init__(self, *, source: str = "noop", message: str = "trading state restore skipped") -> None:
        self._source = source
        self._message = message

    async def restore_trading_state(self) -> RestoreResult:
        logger.info("%s | source=%s", self._message, self._source)
        return RestoreResult(restored=True, source=self._source, message=self._message)


class ExchangeSnapshotStateRestorer:
    """Restore in-memory portfolio from the exchange before trading starts."""

    def __init__(
        self,
        *,
        exchange: Exchange,
        portfolio: PortfolioManager,
        quote_asset: str = "USDT",
    ) -> None:
        self._exchange = exchange
        self._portfolio = portfolio
        self._quote_asset = quote_asset

    async def restore_trading_state(self) -> RestoreResult:
        try:
            if hasattr(self._exchange, "restore_snapshot"):
                snapshot = await self._exchange.restore_snapshot(quote_asset=self._quote_asset)
                balance = float(snapshot.balance)
                positions = list(snapshot.positions)
                open_orders_count = len(snapshot.open_orders)
                recent_executions_count = len(snapshot.recent_executions)
            else:
                balance = await self._exchange.get_balance(self._quote_asset)
                positions = await self._exchange.get_positions()
                open_orders_count = 0
                recent_executions_count = 0
        except Exception:
            logger.exception("Trading state restore failed")
            raise

        self._portfolio.cash[self._quote_asset] = float(balance)
        self._portfolio.positions = list(positions)
        result = RestoreResult(
            restored=True,
            source="exchange_snapshot",
            positions_count=len(positions),
            open_orders_count=open_orders_count,
            recent_executions_count=recent_executions_count,
            message="trading state restored from exchange snapshot",
        )
        logger.info(
            "Trading state restored | source=%s | positions=%s | open_orders=%s | recent_executions=%s | %s_balance=%.8f",
            result.source,
            result.positions_count,
            result.open_orders_count,
            result.recent_executions_count,
            self._quote_asset,
            float(balance),
        )
        return result
