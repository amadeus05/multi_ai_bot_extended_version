import asyncio
import inspect
import logging
import os
import queue
import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable

import numpy as np
import pandas as pd

from core.interfaces.exchange import Exchange
from core.types.commands import CancelOrderCommand, PlaceOrderCommand
from core.types.domain_types import Order, Position
from core.types.enums import OrderSide, OrderStatus
from core.types.events import TradingEvent
from core.types.execution_event_factory import ExecutionEventFactory
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.execution.exit_manager import ExitManager
from domain.portfolio.portfolio_manager import PortfolioManager
from infrastructure.exchanges.bybit.bybit_kline_stream import BybitKlineStream

logger = logging.getLogger(__name__)

# Order.meta: цена закрытия уже включает модельный slippage (например ExitManager / resolve_trade_exit);
# get_order_status не сдвигает avg_price второй раз — иначе paper diverges от бэктеста.
class SimulatedExchange(Exchange):
    """
    Исполнение маркет-ордеров: референсная цена ± slippage, комиссия taker на ногу.
    Round-trip PnL%: вычитается 2 * commission.
    """

    def __init__(
        self,
        commission: float = 0.0004,
        slippage: float = 0.0003,
        leverage: float = 1.0,
        order_id_prefix: str | None = None,
        execution_price_source: Callable[[str, OrderSide], float] | None = None,
        portfolio: PortfolioManager | None = None,
        exit_manager: ExitManager | None = None,
        execution: ExecutionService | None = None,
        exit_ws_url: str | None = None,
        exit_timeframe: str = "1m",
    ):
        self._commission = float(commission)
        self._slippage = float(slippage)
        self._leverage = float(leverage)
        self._orders: dict[str, Order] = {}
        self._order_id = 0
        self._order_id_prefix = order_id_prefix or os.getenv("SIM_ORDER_ID_PREFIX") or f"sim_{uuid.uuid4().hex[:8]}"
        self._execution_price_source = execution_price_source
        self._portfolio = portfolio
        self._exit_manager = exit_manager
        self._execution = execution
        self._exit_timeframe = exit_timeframe
        self._exit_stream = BybitKlineStream(url=exit_ws_url) if exit_ws_url else None
        self._last_price = 0.0

    def set_last_price(self, price: float) -> None:
        p = float(price)
        if np.isfinite(p) and p > 0:
            self._last_price = p

    def market_fill_price(self, reference_price: float, side: OrderSide) -> float:
        """Цена исполнения маркет-ордера от референса (open/mark), как в бэктесте."""
        ref = float(reference_price)
        if not np.isfinite(ref) or ref <= 0:
            return ref
        if side == OrderSide.BUY:
            return float(ref * (1.0 + self._slippage))
        return float(ref * (1.0 - self._slippage))

    async def prepare_market_order(self, order: Order) -> None:
        if self._execution_price_source is None:
            return

        price = self._execution_price_source(order.symbol, order.side)
        if inspect.isawaitable(price):
            price = await price
        price = float(price)
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"invalid simulated execution price: {price!r}")
        order.meta = order.meta or {}
        order.meta.setdefault("signal_price", order.price)
        order.meta["execution_price"] = price
        order.meta["execution_price_refreshed"] = True
        order.price = price

    def net_pnl_pct_round_trip(self, direction: int, entry_price: float, exit_price: float) -> float:
        """direction: 1 long, -1 short. Доходность с вычетом 2 * taker (как бэктест)."""
        e, x = float(entry_price), float(exit_price)
        if int(direction) == 1:
            raw = (x - e) / e
        else:
            raw = (e - x) / e
        return float(raw - 2.0 * self._commission)

    def round_turn_commission_on_notional(self, position_notional: float) -> float:
        return float(position_notional) * (2.0 * self._commission)

    def trade_outcome_from_prices(self, position: dict, exit_price: float) -> tuple[float, float, float]:
        """position: dir, entry, size. -> pnl_pct, pnl_abs, commission."""
        pnl_pct = self.net_pnl_pct_round_trip(int(position["dir"]), float(position["entry"]), float(exit_price))
        notional = float(position["size"])
        commission = self.round_turn_commission_on_notional(notional)
        pnl_abs = notional * pnl_pct
        return float(pnl_pct), float(pnl_abs), float(commission)

    @staticmethod
    def mark_to_market_return_pct(direction: int, entry_price: float, mark_price: float) -> float:
        """Без комиссий — только MTM для эквити."""
        e, m = float(entry_price), float(mark_price)
        if int(direction) == 1:
            return float((m - e) / e)
        return float((e - m) / e)

    async def place_order(self, order: Order) -> str:
        amt = float(order.amount)
        if not np.isfinite(amt) or amt < 0 or amt > 1e18: amt = 0.0
        order.amount = amt
        
        px = float(order.price or self._last_price)
        if not np.isfinite(px) or px <= 0 or px > 1e12: px = self._last_price
        order.price = px
        
        self._order_id += 1
        order_id = f"{self._order_id_prefix}_{self._order_id}"
        self._orders[order_id] = order
        return order_id

    async def get_order_status(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if not order: return {"status": OrderStatus.REJECTED}
        
        ref_px = float(order.price or self._last_price)
        if not np.isfinite(ref_px) or ref_px <= 0: ref_px = 1.0

        meta = getattr(order, "meta", None) or {}
        if meta.get(ORDER_META_FILL_PRICE_FINAL):
            avg_px = ref_px
        else:
            avg_px = self.market_fill_price(ref_px, order.side)
        if not np.isfinite(avg_px):
            avg_px = ref_px
        
        fee = abs(float(order.amount) * avg_px) * self._commission
        if not np.isfinite(fee): fee = 0.0
        
        liq = self._calculate_liquidation_price(order.side, avg_px)
        return {
            "status": OrderStatus.FILLED,
            "filled": order.amount,
            "avg_price": avg_px,
            "fee": fee,
            "liquidation_price": liq,
            "leverage": self._leverage,
        }

    async def submit_order_lifecycle(self, command: PlaceOrderCommand) -> list[TradingEvent]:
        order = command.order
        order_id = await self.place_order(order)
        order.id = order_id
        status = await self.get_order_status(order_id)
        event_ts = ExecutionEventFactory.event_ts(command.ts, status)
        status_value = status.get("status")

        if ExecutionEventFactory.is_status(status_value, OrderStatus.REJECTED):
            return [ExecutionEventFactory.rejected(event_ts, order, str(status.get("reason", "rejected")))]

        if ExecutionEventFactory.is_status(status_value, OrderStatus.CANCELLED):
            return [
                ExecutionEventFactory.cancelled(
                    event_ts,
                    order_id=order_id,
                    client_order_id=order.client_order_id,
                    reason=str(status.get("reason", "cancelled")),
                )
            ]

        events: list[TradingEvent] = [
            ExecutionEventFactory.accepted(event_ts, order_id, order)
        ]
        if not ExecutionEventFactory.is_status(status_value, OrderStatus.FILLED):
            return events

        events.append(ExecutionEventFactory.fill_from_status(command, order_id, status, event_ts))
        return events

    async def stream_execution_events(self) -> AsyncIterator[TradingEvent]:
        if self._portfolio is None or self._exit_manager is None or self._execution is None or self._exit_stream is None:
            return

        stream = self._exit_stream
        loop = asyncio.get_running_loop()
        subscribed_symbols: set[str] = set()
        stream.start()
        connected = await loop.run_in_executor(None, stream.wait_until_connected, 15.0)
        if not connected:
            stream.stop()
            raise RuntimeError("Paper 1m exit websocket connection timeout")

        try:
            while True:
                open_symbols = {position.symbol for position in self._portfolio.get_open_positions()}
                if open_symbols != subscribed_symbols:
                    topics = {
                        f"kline.{self._exit_interval()}.{symbol.replace('/', '').replace('-', '').upper()}"
                        for symbol in open_symbols
                    }
                    stream.sync_topics(topics)
                    subscribed_symbols = open_symbols
                    logger.info("Paper 1m exit monitor topics synced | symbols=%s", ",".join(sorted(open_symbols)) or "-")

                try:
                    event = await loop.run_in_executor(None, stream.get_event, 1.0)
                except queue.Empty:
                    continue

                symbol = self._from_api_symbol(event.symbol)
                position = self._portfolio.get_position(symbol)
                if position is None:
                    continue

                fill_event = await self._exit_event_for_kline(position, event)
                if fill_event is not None:
                    yield fill_event
        finally:
            stream.stop()

    def _exit_interval(self) -> str:
        return {"1m": "1", "3m": "3", "5m": "5"}.get(self._exit_timeframe, self._exit_timeframe)

    @staticmethod
    def _from_api_symbol(symbol: str) -> str:
        upper = str(symbol).upper()
        if upper.endswith("USDT") and "/" not in upper:
            return f"{upper[:-4]}/USDT"
        return upper

    async def _exit_event_for_kline(self, position: Position, event) -> TradingEvent | None:
        meta = position.meta or {}
        if meta.get("barrier_stop_pct") is None or meta.get("barrier_take_pct") is None:
            return None

        exit_price, reason = self._exit_manager.check_causal_exit(
            position=position,
            next_open=float(event.open_price),
            next_high=float(event.high_price),
            next_low=float(event.low_price),
            stop_pct=float(meta["barrier_stop_pct"]),
            take_pct=float(meta["barrier_take_pct"]),
        )
        if exit_price is None:
            return None

        side = OrderSide.SELL if position.side.value == "long" else OrderSide.BUY
        order = Order(
            symbol=position.symbol,
            side=side,
            amount=float(position.amount),
            price=float(exit_price),
            meta={
                ORDER_META_FILL_PRICE_FINAL: True,
                "_engine_exit_reason": reason,
                "_engine_reason_suffix": "1M",
            },
        )
        command = PlaceOrderCommand(
            order,
            reason=reason,
            ts=pd.to_datetime(event.end_ms, unit="ms", utc=True).tz_convert(None),
        )
        events = (
            await self._execution.execute(command, self)
            if self._execution is not None
            else await self.submit_order_lifecycle(command)
        )
        for item in events:
            if getattr(item, "symbol", None) == position.symbol and item.__class__.__name__ == "FillEvent":
                logger.info(
                    "[%s] paper 1m exit hit | reason=%s | price=%.8f",
                    position.symbol,
                    reason,
                    float(exit_price),
                )
                return item
        return None

    async def cancel_order_lifecycle(self, command: CancelOrderCommand) -> list[TradingEvent]:
        event_ts = ExecutionEventFactory.event_ts(command.ts)
        cancelled = await self.cancel_order(command.order_id)
        if cancelled is False:
            return [ExecutionEventFactory.cancel_rejected(command, event_ts)]
        return [
            ExecutionEventFactory.cancelled(
                event_ts,
                command.order_id,
                reason=command.reason,
            )
        ]

    def _calculate_liquidation_price(self, side: OrderSide, entry_price: float) -> float:
        mmr = 0.005
        lev = float(self._leverage)
        if lev <= 0 or not np.isfinite(lev): lev = 1.0
        try:
            if side == OrderSide.BUY:
                liq = entry_price * (1.0 - (1.0 / lev) + mmr)
            else:
                liq = entry_price * (1.0 + (1.0 / lev) - mmr)
            return float(liq) if np.isfinite(liq) else 0.0
        except: return 0.0

    async def current_price(self, symbol: str) -> float:
        return self._last_price

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            return True
        return False

    async def get_balance(self, asset: str) -> float:
        return 0.0

    async def get_positions(self) -> list[Position]:
        return []
