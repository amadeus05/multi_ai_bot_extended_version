import numpy as np

from core.interfaces.exchange import Exchange
from core.types.domain_types import Order, Position
from core.types.enums import OrderSide, OrderStatus
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL

# Order.meta: цена закрытия уже включает модельный slippage (например ExitManager / resolve_trade_exit);
# get_order_status не сдвигает avg_price второй раз — иначе paper diverges от бэктеста.
class SimulatedExchange(Exchange):
    """
    Исполнение маркет-ордеров: референсная цена ± slippage, комиссия taker на ногу.
    Round-trip PnL%: вычитается 2 * commission.
    """

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

    def market_fill_price(self, reference_price: float, side: OrderSide) -> float:
        """Цена исполнения маркет-ордера от референса (open/mark), как в бэктесте."""
        ref = float(reference_price)
        if not np.isfinite(ref) or ref <= 0:
            return ref
        if side == OrderSide.BUY:
            return float(ref * (1.0 + self._slippage))
        return float(ref * (1.0 - self._slippage))

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
        order_id = f"sim_{self._order_id}"
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
