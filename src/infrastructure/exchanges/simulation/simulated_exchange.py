import os
import numpy as np

from core.interfaces.exchange import Exchange
from core.types.domain_types import Order, Position, Trade
from core.types.enums import OrderSide, OrderStatus


class SimulatedExchange(Exchange):
    def __init__(self, commission: float = 0.0004, slippage: float = 0.0003, leverage: float = 1.0):
        self._commission = float(commission)
        self._slippage = float(slippage)
        self._leverage = float(leverage)
        self._orders: dict[str, Order] = {}
        self._order_id = 0
        self._last_price = 0.0

    def set_last_price(self, price: float) -> None:
        p = float(price)
        if np.isfinite(p) and p > 0:
            self._last_price = p

    async def place_order(self, order: Order) -> str:
        amt = float(order.amount)
        if not np.isfinite(amt) or amt < 0 or amt > 1e18: amt = 0.0
        order.amount = amt
        
        px = float(order.price or self._last_price)
        if not np.isfinite(px) or px <= 0 or px > 1e12: px = self._last_price
        order.price = px
        
        self._order_id += 1
        order_id = f"sim_{self._order_id}"
        self._orders[order_id] = order
        return order_id

    async def get_order_status(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if not order: return {"status": OrderStatus.REJECTED}
        
        ref_px = float(order.price or self._last_price)
        if not np.isfinite(ref_px) or ref_px <= 0: ref_px = 1.0
        
        if order.side == OrderSide.BUY:
            avg_px = ref_px * (1.0 + self._slippage)
        else:
            avg_px = ref_px * (1.0 - self._slippage)
            
        if not np.isfinite(avg_px): avg_px = ref_px
        
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
