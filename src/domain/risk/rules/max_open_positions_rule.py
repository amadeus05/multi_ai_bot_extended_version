from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule


class MaxOpenPositionsRule(RiskRule):
    def apply(self, ctx: RiskContext) -> bool:
        position = ctx.portfolio.get_position(ctx.order.symbol)
        if position is not None:
            # Allow managing/reducing an existing position even when open-position cap is reached.
            return True
        return len(ctx.portfolio.positions) < ctx.profile.max_open_positions
