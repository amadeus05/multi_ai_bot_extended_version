from core.types.enums import OrderSide, PositionSide
from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule
from domain.risk.sizing import barrier_nominal_under_margin_cap, estimated_margin_reserved_perps_linear


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

        total_cash = float(ctx.portfolio.cash.get(ctx.quote_asset, 0.0))
        if total_cash <= 0:
            return False

        opens = getattr(ctx.portfolio, "get_open_positions", lambda: [])()
        reserved = estimated_margin_reserved_perps_linear(opens, ctx.profile.leverage)
        available_margin = max(0.0, total_cash - reserved)

        effective_risk = ctx.profile.risk_per_trade
        if (
            ctx.consecutive_losses >= ctx.profile.reduce_risk_after_consecutive_losses
            and float(ctx.profile.reduced_risk_per_trade) > 0
        ):
            effective_risk = ctx.profile.reduced_risk_per_trade

        position_notional, required_margin = barrier_nominal_under_margin_cap(
            total_cash,
            available_margin,
            effective_risk,
            sp,
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
