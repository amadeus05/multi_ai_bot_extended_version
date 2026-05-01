from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule


class MaxNewPositionsPerBarRule(RiskRule):
    def apply(self, ctx: RiskContext) -> bool:
        position = ctx.portfolio.get_position(ctx.order.symbol)
        if position is not None:
            # Allow closes/reductions inside current bar.
            return True
        return ctx.new_positions_in_bar < ctx.profile.max_new_positions_per_bar
