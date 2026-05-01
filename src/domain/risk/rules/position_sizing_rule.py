from core.types.enums import OrderSide, PositionSide
from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule


class PositionSizingRule(RiskRule):
    def apply(self, ctx: RiskContext) -> bool:
        price = float(ctx.order.price or 0.0)
        if price <= 0:
            return False

        position = ctx.portfolio.get_position(ctx.order.symbol)
        if position is not None:
            is_reducing = (
                (position.side == PositionSide.LONG and ctx.order.side == OrderSide.SELL)
                or (position.side == PositionSide.SHORT and ctx.order.side == OrderSide.BUY)
            )
            if is_reducing:
                # Keep explicit close/reduce size from strategy; never upsize from risk sizing.
                ctx.order.amount = min(float(ctx.order.amount), float(position.amount))
                return ctx.order.amount > 0

        available_cash = float(ctx.portfolio.cash.get(ctx.quote_asset, 0.0))
        if available_cash <= 0:
            return False

        effective_risk = ctx.profile.risk_per_trade
        if (
            ctx.consecutive_losses >= ctx.profile.reduce_risk_after_consecutive_losses
            and float(ctx.profile.reduced_risk_per_trade) > 0
        ):
            effective_risk = ctx.profile.reduced_risk_per_trade

        target_notional = available_cash * effective_risk * ctx.profile.leverage
        if target_notional <= 0:
            return False

        amount = target_notional / price
        if amount <= 0 or amount != amount:
            return False

        ctx.order.amount = amount
        return True
