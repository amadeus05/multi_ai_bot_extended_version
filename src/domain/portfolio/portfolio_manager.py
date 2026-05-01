from dataclasses import dataclass, field

from core.types.enums import OrderSide, PositionSide
from core.types.domain_types import Position, Trade


@dataclass
class PortfolioManager:
    cash: dict[str, float]
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    closed_trade_results: list[dict] = field(default_factory=list)
    trade_events: list[dict] = field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    _trade_seq: int = 0
    _active_trade_number_by_symbol: dict[str, int] = field(default_factory=dict)
    _entry_fee_pool_by_symbol: dict[str, float] = field(default_factory=dict)

    def apply_execution(self, execution: Trade) -> None:
        self._apply_commission(execution.fee)
        self._apply_position_update(execution)
        self.trades.append(execution)

    def get_position(self, symbol: str) -> Position | None:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    def get_open_positions(self) -> list[Position]:
        return list(self.positions)

    def get_state_snapshot(self) -> dict:
        return {
            "cash": dict(self.cash),
            "positions": [position for position in self.positions],
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "trades_count": len(self.trades),
            "closed_trades_count": len(self.closed_trade_results),
        }

    def apply_fee(self, amount: float, quote_asset: str = "USDT") -> None:
        amount = float(amount)
        if not np.isfinite(amount):
            print(f"WARNING: Non-finite fee amount: {amount}")
            return
        self.cash[quote_asset] = float(self.cash.get(quote_asset, 0.0)) - amount

    def _apply_commission(self, fee: float, quote_asset: str = "USDT") -> None:
        self.apply_fee(fee, quote_asset=quote_asset)

    def _apply_position_update(self, trade: Trade) -> None:
        qty = float(trade.amount)
        if qty <= 0:
            return
        sign = 1.0 if trade.side == OrderSide.BUY else -1.0
        signed_qty = sign * qty

        current = self.get_position(trade.symbol)
        if current is None:
            side = PositionSide.LONG if signed_qty > 0 else PositionSide.SHORT
            self.positions.append(Position(symbol=trade.symbol, side=side, amount=abs(signed_qty), entry_price=trade.price, meta=trade.meta.copy() if trade.meta else None))
            self._trade_seq += 1
            trade_number = self._trade_seq
            self._active_trade_number_by_symbol[trade.symbol] = trade_number
            self._entry_fee_pool_by_symbol[trade.symbol] = float(trade.fee)
            open_ev = {
                "type": "OPEN",
                "trade_number": trade_number,
                "symbol": trade.symbol,
                "side": side.value.upper(),
                "price": float(trade.price),
                "qty": float(abs(signed_qty)),
                "fee": float(trade.fee),
                "ts": trade.ts,
                "balance": float(self.cash.get("USDT", 0.0)),
            }
            if getattr(trade, "meta", None):
                for k in ("p_long", "p_short", "score", "signal_gap", "direction_prob"):
                    if k in trade.meta:
                        open_ev[k] = trade.meta[k]
            self.trade_events.append(open_ev)
            return

        current_signed_qty = current.amount if current.side == PositionSide.LONG else -current.amount
        new_signed_qty = current_signed_qty + signed_qty

        if current_signed_qty * signed_qty < 0:
            closed_qty = min(abs(current_signed_qty), abs(signed_qty))
            current_qty_abs = abs(current_signed_qty)
            executed_qty_abs = abs(signed_qty)
            pnl_per_unit = (trade.price - current.entry_price) if current.side == PositionSide.LONG else (current.entry_price - trade.price)
            realized_chunk = closed_qty * pnl_per_unit
            self.realized_pnl += realized_chunk
            self.cash[trade.quote_asset if hasattr(trade, 'quote_asset') else "USDT"] += realized_chunk
            pnl_pct = (pnl_per_unit / current.entry_price) if current.entry_price else 0.0
            trade_number = int(self._active_trade_number_by_symbol.get(trade.symbol, 0))
            entry_fee_pool = float(self._entry_fee_pool_by_symbol.get(trade.symbol, 0.0))
            entry_fee_alloc = entry_fee_pool * (closed_qty / current_qty_abs) if current_qty_abs > 0 else 0.0
            exit_fee_alloc = float(trade.fee) * (closed_qty / executed_qty_abs) if executed_qty_abs > 0 else 0.0
            commission_round_trip = entry_fee_alloc + exit_fee_alloc
            self.closed_trade_results.append(
                {
                    "trade_number": trade_number,
                    "symbol": trade.symbol,
                    "side": current.side.value,
                    "entry_price": float(current.entry_price),
                    "exit_price": float(trade.price),
                    "qty": float(closed_qty),
                    "pnl_pct": float(pnl_pct),
                    "pnl_abs_gross": float(realized_chunk),
                    "pnl_abs": float(realized_chunk - commission_round_trip),
                    "commission": float(commission_round_trip),
                    "ts": trade.ts,
                }
            )
            remaining_entry_fee_pool = max(0.0, entry_fee_pool - entry_fee_alloc)
            self._entry_fee_pool_by_symbol[trade.symbol] = remaining_entry_fee_pool
            self.trade_events.append(
                {
                    "type": "CLOSE",
                    "trade_number": trade_number,
                    "symbol": trade.symbol,
                    "side": current.side.value.upper(),
                    "reason": "CLOSE",
                    "pnl_pct": float(pnl_pct),
                    "pnl_abs": float(realized_chunk - commission_round_trip),
                    "commission": float(commission_round_trip),
                    "ts": trade.ts,
                    "balance": float(self.cash.get("USDT", 0.0)),
                }
            )

        if new_signed_qty == 0:
            self.positions = [position for position in self.positions if position.symbol != trade.symbol]
            self._active_trade_number_by_symbol.pop(trade.symbol, None)
            self._entry_fee_pool_by_symbol.pop(trade.symbol, None)
            return

        if current_signed_qty * signed_qty > 0:
            total_qty = abs(current_signed_qty) + abs(signed_qty)
            weighted_entry = ((abs(current_signed_qty) * current.entry_price) + (abs(signed_qty) * trade.price)) / total_qty
            current.entry_price = weighted_entry
            current.amount = abs(new_signed_qty)
            current.side = PositionSide.LONG if new_signed_qty > 0 else PositionSide.SHORT
            if trade.meta:
                current.meta = current.meta or {}
                current.meta.update(trade.meta)
            return

        if abs(signed_qty) > abs(current_signed_qty):
            current.entry_price = trade.price
            self._trade_seq += 1
            self._active_trade_number_by_symbol[trade.symbol] = self._trade_seq
            executed_qty_abs = abs(signed_qty)
            flip_open_qty = max(0.0, executed_qty_abs - abs(current_signed_qty))
            flip_open_fee = float(trade.fee) * (flip_open_qty / executed_qty_abs) if executed_qty_abs > 0 else 0.0
            self._entry_fee_pool_by_symbol[trade.symbol] = float(flip_open_fee)
            if trade.meta:
                current.meta = trade.meta.copy()
            else:
                current.meta = None
        current.amount = abs(new_signed_qty)
        current.side = PositionSide.LONG if new_signed_qty > 0 else PositionSide.SHORT

    async def revalue_positions(self, market_prices: dict[str, float], quote_asset: str = "USDT") -> float:
        total = self.cash.get(quote_asset, 0.0)
        unrealized = 0.0
        for position in self.positions:
            px = float(market_prices.get(position.symbol, position.entry_price))
            if not np.isfinite(px):
                print(f"CRITICAL: Non-finite price for {position.symbol}: {px}")
                px = position.entry_price
            signed_qty = position.amount if position.side == PositionSide.LONG else -position.amount
            total += signed_qty * px
            unrealized += signed_qty * (px - position.entry_price)
        if not np.isfinite(total):
            print(f"CRITICAL: Non-finite total equity calculated: {total}")
            total = self.cash.get(quote_asset, 0.0)
        self.unrealized_pnl = unrealized
        return total

    async def equity(self, exchange: "Exchange", quote_asset: str = "USDT") -> float:
        total = self.cash.get(quote_asset, 0.0)
        for pos in self.positions:
            px = await exchange.current_price(pos.symbol)
            total += pos.amount * px
        return total


# Чего не хватает до “боевого”:

# Нет полноценного учета partial fills и нескольких fill-ов одного ордера.
# Нет строгой синхронизации с биржевым источником истины (reconciliation/repair).
# Нет lock/atomicity для конкурентных обновлений (параллельные тики/исполнения).
# Нет idempotency по execution events (защита от дублей).
# Нет маржинальной модели (isolated/cross, funding, borrow, maintenance margin, liquidation metrics).
# Нет корпоративных actions по деривативам/перпам (funding accrual по времени).
# Нет event-sourcing/audit trail уровня “почему состояние такое”.
# Нет тестов на крайние кейсы (flip позиции, over-close, out-of-order fills, duplicate fills).