from __future__ import annotations

from application.trading_engine import ENGINE_META_EXIT_REASON, TradingEngine
from core.types.commands import PlaceOrderCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.portfolio.portfolio_manager import PortfolioManager


async def close_all_positions_at_market(
    *,
    engine: TradingEngine,
    portfolio: PortfolioManager,
    exchange,
    market_prices: dict[str, float],
    ts,
) -> None:
    commands: list[PlaceOrderCommand] = []
    for pos in list(portfolio.get_open_positions()):
        ref_price = float(market_prices.get(pos.symbol, pos.entry_price))
        side = OrderSide.SELL if pos.side.value == "long" else OrderSide.BUY
        if hasattr(exchange, "market_fill_price"):
            exit_price = exchange.market_fill_price(ref_price, side)
        else:
            exit_price = ref_price
        order = Order(
            symbol=pos.symbol,
            side=side,
            amount=pos.amount,
            price=exit_price,
            meta={
                ORDER_META_FILL_PRICE_FINAL: True,
                "reason": "FINAL",
                ENGINE_META_EXIT_REASON: "FINAL",
            },
        )
        commands.append(PlaceOrderCommand(order, reason="FINAL", ts=ts))

    if commands:
        await engine.execute_commands(commands)
