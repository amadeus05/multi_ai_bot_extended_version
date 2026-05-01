from domain.risk.models.risk_context import RiskContext
from domain.risk.rules.base_rule import RiskRule


class MaxSLPerDayRule(RiskRule):
    def apply(self, ctx: RiskContext) -> bool:
        if int(ctx.profile.max_sl_per_day) <= 0:
            return True
        day_key = str(getattr(ctx.current_bar_ts, "date", lambda: "unknown")())
        return ctx.sl_per_day.get(day_key, 0) < ctx.profile.max_sl_per_day
