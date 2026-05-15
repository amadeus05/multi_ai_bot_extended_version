import pandas as pd

from core.types.domain_types import Order
from domain.risk.models.risk_context import RiskContext
from domain.risk.models.risk_profile import RiskProfile
from domain.risk.risk_manager import RiskManager
from domain.risk.rules.max_new_positions_per_bar_rule import MaxNewPositionsPerBarRule
from domain.risk.rules.max_open_positions_rule import MaxOpenPositionsRule
from domain.risk.rules.max_sl_per_day_rule import MaxSLPerDayRule
from domain.risk.rules.position_sizing_rule import PositionSizingRule
from domain.risk.rules.sl_cooldown_rule import SLCooldownRule


class AdaptiveRiskManager(RiskManager):
    def __init__(self, profile: RiskProfile, quote_asset: str = "USDT") -> None:
        self.profile = profile
        self.quote_asset = quote_asset
        self._current_bar_ts = None
        self._bar_index = 0
        self._new_positions_in_bar = 0
        self._symbol_cooldown_until_bar: dict[str, int] = {}
        self._sl_per_day: dict[str, int] = {}
        self._consecutive_losses = 0
        self._pre_rules = [
            MaxSLPerDayRule(),
            MaxOpenPositionsRule(),
            MaxNewPositionsPerBarRule(),
            SLCooldownRule(),
        ]
        self._sizing_rule = PositionSizingRule()

    def set_current_bar(self, bar_ts) -> None:
        if self._current_bar_ts == bar_ts:
            return
        self._current_bar_ts = bar_ts
        self._bar_index += 1
        self._new_positions_in_bar = 0

    def register_trade_result(self, symbol: str, pnl: float, bar_ts, stop_loss_hit: bool = False) -> None:
        day_key = self._day_key(bar_ts)
        if stop_loss_hit:
            self._sl_per_day[day_key] = self._sl_per_day.get(day_key, 0) + 1
            self._symbol_cooldown_until_bar[symbol] = self._bar_index + self.profile.sl_cooldown_bars

        prev_losses = self._consecutive_losses
        if pnl < 0:
            self._consecutive_losses += 1
            if self._risk_reduction_enabled() and prev_losses < self.profile.reduce_risk_after_consecutive_losses <= self._consecutive_losses:
                print(
                    f"[{bar_ts}] ⚠️ Loss streak {self._consecutive_losses}: "
                    f"risk per trade reduced to {self._fmt_risk_pct(self.profile.reduced_risk_per_trade)}"
                )
        else:
            if self._risk_reduction_enabled() and prev_losses >= self.profile.reduce_risk_after_consecutive_losses:
                print(
                    f"[{bar_ts}] ℹ️ Loss streak reset: "
                    f"risk per trade restored to {self._fmt_risk_pct(self.profile.risk_per_trade)}"
                )
            self._consecutive_losses = 0

    @staticmethod
    def _day_key(value) -> str:
        try:
            return str(pd.Timestamp(value).date())
        except Exception:
            return str(getattr(value, "date", lambda: value)())

    def restore_from_closed_trades(self, closed_trades: list[dict]) -> None:
        """Seed restart-sensitive risk counters from persisted closed trades."""
        self._sl_per_day = {}
        self._consecutive_losses = 0

        for row in closed_trades:
            reason = str(row.get("reason") or row.get("exit_reason") or "").upper()
            ts = row.get("ts") or row.get("exit_time")
            if reason == "SL":
                day_key = self._day_key(ts)
                self._sl_per_day[day_key] = self._sl_per_day.get(day_key, 0) + 1

            try:
                pnl = float(row.get("pnl_abs", row.get("pnl", 0.0)) or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def _risk_reduction_enabled(self) -> bool:
        return (
            int(self.profile.reduce_risk_after_consecutive_losses) > 0
            and float(self.profile.reduced_risk_per_trade) > 0
            and float(self.profile.reduced_risk_per_trade) != float(self.profile.risk_per_trade)
        )

    @staticmethod
    def _fmt_risk_pct(value: float) -> str:
        return f"{float(value) * 100.0:.2f}%"

    def check(self, order: Order, portfolio, exchange):
        ctx = RiskContext(
            order=order,
            portfolio=portfolio,
            exchange=exchange,
            profile=self.profile,
            quote_asset=self.quote_asset,
            bar_index=self._bar_index,
            current_bar_ts=self._current_bar_ts,
            sl_per_day=self._sl_per_day,
            symbol_cooldown_until_bar=self._symbol_cooldown_until_bar,
            new_positions_in_bar=self._new_positions_in_bar,
            consecutive_losses=self._consecutive_losses,
        )

        for rule in self._pre_rules:
            if not rule.apply(ctx):
                return None

        if not self._sizing_rule.apply(ctx):
            return None

        self._new_positions_in_bar += 1
        return order



# пока не боевой.

# Сейчас это хороший каркас + базовые ограничения, но для прод-режима не хватает критичных вещей:

# Нет реального учета PnL/закрытий в контуре (метод register_trade_result(...) есть, но не встроен в полный lifecycle исполнения).
# Нет гарантированного источника истинной позиции/баланса с биржи перед risk-check (сейчас ориентир в основном на локальный state).
# Нет валидации биржевых ограничений (min_qty, step_size, tick_size, min_notional) по каждому символу.
# Нет kill-switch по equity drawdown / дневному лимиту потерь в деньгах.
# Нет защиты от гонок/дубликатов при параллельных тиках и повторных ордерах (idempotency на уровне risk+execution).
# Нет режимов деградации при отказах API/рассинхроне состояния.
# Нет тестового покрытия (unit + scenario + stress) для риск-правил.
# То есть: для backtest/dev — ок, для live с деньгами — пока рано.

# Если хочешь, могу довести до “боевого минимального” в следующем шаге:

# связать register_trade_result с фактическими закрытиями и стопами,
# добавить exchange filters и rounding,
# сделать hard kill-switch по drawdown/day loss,
# закрыть это тестами на ключевые сценарии.
