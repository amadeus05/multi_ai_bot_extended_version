from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule


class SLCooldownRule(RiskRule):
    def apply(self, ctx: RiskContext) -> bool:
        cooldown_bar = ctx.symbol_cooldown_until_bar.get(ctx.order.symbol, -1)
        return cooldown_bar <= ctx.bar_index
