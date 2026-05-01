from core.types.enums import OrderSide, PositionSide
from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule
from domain.risk.sizing import (
    cap_notional_to_available_margin,
    compute_barrier_position_notional,
)


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

        meta = getattr(ctx.order, "meta", None) or {}
        stop_pct = meta.get("barrier_stop_pct")
        try:
            sp = float(stop_pct) if stop_pct is not None else None
        except (TypeError, ValueError):
            sp = None
        if sp is None or sp <= 0 or not (sp == sp):
            return False

        balance_for_risk = float(ctx.portfolio.cash.get(ctx.quote_asset, 0.0))
        if balance_for_risk <= 0:
            return False

        effective_risk = ctx.profile.risk_per_trade
        if (
            ctx.consecutive_losses >= ctx.profile.reduce_risk_after_consecutive_losses
            and float(ctx.profile.reduced_risk_per_trade) > 0
        ):
            effective_risk = ctx.profile.reduced_risk_per_trade

        position_notional, required_margin = compute_barrier_position_notional(
            balance_for_risk,
            effective_risk,
            sp,
            ctx.profile.leverage,
        )
        position_notional, required_margin = cap_notional_to_available_margin(
            position_notional,
            required_margin,
            balance_for_risk,
            ctx.profile.leverage,
        )
        if position_notional < float(ctx.profile.min_position_notional):
            return False
        if required_margin <= 0 or position_notional <= 0:
            return False

        amount = position_notional / price
        if amount <= 0 or amount != amount:
            return False

        ctx.order.amount = amount
        return True
