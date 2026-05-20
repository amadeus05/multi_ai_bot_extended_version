import logging
from dataclasses import dataclass

from application.event_journal import EventJournal
from application.execution_event_deduplicator import ExecutionEventDeduplicator
from application.trading_notifications import TradingNotificationFactory
from core.interfaces.data_provider import DataProvider
from core.interfaces.execution_lifecycle import MarketOrderPreparationExchange
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from core.interfaces.notifier import Notifier
from core.types.commands import CancelOrderCommand, PlaceOrderCommand, TradingCommand
from core.types.domain_types import Order
from core.types.enums import OrderSide
from core.types.events import (
    FillEvent,
    MarketEvent,
    OrderAcceptedEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
    TradingEvent,
)
from core.types.notifications import SignalNotification, TradeExitNotification
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.risk_manager import RiskManager
from domain.ml.labeling.barrier_policy import BarrierPolicy
from domain.ml.labeling.models import LabelingConfig
from domain.strategy.strategy_engine import Strategy


ENGINE_META_EXIT_REASON = "_engine_exit_reason"
ENGINE_META_REASON_SUFFIX = "_engine_reason_suffix"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EntryCandidate:
    tick: object
    order: Order
    prediction: dict
    features: object
    score: float


class TradingEngine:
    def __init__(
        self,
        exchange: Exchange,
        data_provider: DataProvider,
        model: Model,
        strategy: Strategy,
        risk_manager: RiskManager,
        portfolio: PortfolioManager,
        execution: ExecutionService,
        exit_manager=None,
        barrier_policy: BarrierPolicy | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.exit_manager = exit_manager
        self._barrier_policy = barrier_policy or BarrierPolicy(LabelingConfig.from_env())
        self._deps = {
            "exchange": exchange,
            "data": data_provider,
            "model": model,
            "strategy": strategy,
            "risk": risk_manager,
            "portfolio": portfolio,
            "execution": execution,
        }
        self._running = False
        self._execution_dedupe = ExecutionEventDeduplicator()
        self._event_journal: EventJournal | None = None
        self._notifier = notifier
        self._signal_seq = 0
        self._notifications = TradingNotificationFactory()

    def set_event_journal(self, journal: EventJournal | None) -> None:
        self._event_journal = journal

    def restore_risk_history(self, closed_trades: list[dict]) -> None:
        risk = self._deps["risk"]
        if hasattr(risk, "restore_from_closed_trades"):
            risk.restore_from_closed_trades(closed_trades)

    @staticmethod
    def _tick_price(tick, name: str, fallback: float) -> float:
        value = getattr(tick, name, None)
        if value is None:
            return float(fallback)
        return float(value)

    def _prepare_tick_context(self, tick) -> None:
        risk = self._deps["risk"]
        exchange = self._deps["exchange"]
        if hasattr(exchange, "set_last_price"):
            exchange.set_last_price(float(tick.price))
        if hasattr(risk, "set_current_bar"):
            risk.set_current_bar(getattr(tick, "ts", None))

    async def _notify_signal(self, notification: SignalNotification) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.notify_signal(notification)
        except Exception:
            logger.exception("Signal notification failed")

    async def _notify_trade_exit(self, notification: TradeExitNotification) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.notify_trade_exit(notification)
        except Exception:
            logger.exception("Trade exit notification failed")

    def _notifier_wants_signal_charts(self) -> bool:
        return bool(getattr(self._notifier, "wants_signal_charts", False))

    def _fill_meta_reason(self, fill: FillEvent) -> tuple[str | None, str | None]:
        meta = fill.meta or {}
        reason = meta.get(ENGINE_META_EXIT_REASON) or meta.get("reason")
        suffix = meta.get(ENGINE_META_REASON_SUFFIX)
        return reason, suffix

    async def _apply_fill(self, fill: FillEvent) -> None:
        risk = self._deps["risk"]
        portfolio = self._deps["portfolio"]
        trade = fill.to_trade()

        prev_closed = len(portfolio.closed_trade_results)
        portfolio.apply_execution(trade)

        reason, reason_suffix = self._fill_meta_reason(fill)
        if reason:
            for res in portfolio.closed_trade_results[prev_closed:]:
                res["reason"] = reason
                is_sl = reason == "SL"
                if hasattr(risk, "register_trade_result"):
                    risk.register_trade_result(res["symbol"], res["pnl_abs"], fill.ts, stop_loss_hit=is_sl)
            if portfolio.trade_events and portfolio.trade_events[-1].get("type") == "CLOSE":
                portfolio.trade_events[-1]["reason"] = reason
                if reason_suffix:
                    portfolio.trade_events[-1]["reason_suffix"] = reason_suffix

        for res in portfolio.closed_trade_results[prev_closed:]:
            await self._notify_trade_exit(self._trade_exit_notification(res, portfolio))

    async def _signal_notification(
        self,
        *,
        tick,
        order: Order,
        portfolio: PortfolioManager,
        features,
    ) -> SignalNotification:
        return await self._notifications.signal(
            tick=tick,
            order=order,
            portfolio=portfolio,
            features=features,
            signal_seq=self._signal_seq,
            include_chart=self._notifier_wants_signal_charts(),
        )

    def _trade_exit_notification(self, res: dict, portfolio: PortfolioManager) -> TradeExitNotification:
        return self._notifications.trade_exit(res, portfolio)

    async def _commands_for_exit_tick(
        self,
        tick,
        *,
        reason_suffix: str | None = None,
        continue_with_entry: bool = False,
    ) -> list[TradingCommand]:
        portfolio = self._deps["portfolio"]
        if self.exit_manager is not None:
            pos = portfolio.get_position(tick.symbol)
            if pos is not None:
                exit_price, reason = self.exit_manager.check_causal_exit(
                    position=pos,
                    next_open=self._tick_price(tick, "open", tick.price),
                    next_high=self._tick_price(tick, "high", tick.price),
                    next_low=self._tick_price(tick, "low", tick.price),
                    stop_pct=pos.meta.get("barrier_stop_pct", float('inf')) if pos.meta else float('inf'),
                    take_pct=pos.meta.get("barrier_take_pct", float('inf')) if pos.meta else float('inf')
                )
                if exit_price is not None:
                    from core.types.enums import OrderSide
                    exit_side = OrderSide.SELL if pos.side.value == "long" else OrderSide.BUY
                    from core.types.domain_types import Order
                    exit_order = Order(
                        symbol=pos.symbol,
                        side=exit_side,
                        amount=pos.amount,
                        price=exit_price,
                        meta={
                            ORDER_META_FILL_PRICE_FINAL: True,
                            ENGINE_META_EXIT_REASON: reason,
                            ENGINE_META_REASON_SUFFIX: reason_suffix,
                        },
                    )
                    return [
                        PlaceOrderCommand(
                            exit_order,
                            reason=reason,
                            ts=getattr(tick, "ts", None),
                            source_tick=tick if continue_with_entry else None,
                            continue_with_entry=continue_with_entry,
                        )
                    ]
        return []

    async def _entry_candidate_for_tick(self, tick) -> _EntryCandidate | None:
        portfolio = self._deps["portfolio"]

        if portfolio.get_position(tick.symbol) is not None:
            logger.info(
                "scan result | symbol=%s | ts=%s | price=%s | decision=skip_existing_position",
                tick.symbol,
                getattr(tick, "ts", None),
                getattr(tick, "price", None),
            )
            return None

        df = await self._deps["data"].warmup(tick.symbol, self._deps["model"].required_bars())
        prediction = self._deps["model"].predict(df)
        bp = self._barrier_policy.barriers_for_last_row(df)
        if bp is not None:
            prediction["barrier_stop_pct"], prediction["barrier_take_pct"] = bp
        raw_order = self._deps["strategy"].on_prediction(tick, prediction, self._deps["portfolio"])
        p_long = float(prediction.get("p_long", 0.5))
        p_short = float(prediction.get("p_short", 0.5))
        signal_gap = abs(p_long - p_short)
        if raw_order is None:
            logger.info(
                "scan result | symbol=%s | ts=%s | price=%s | p_long=%.4f | p_short=%.4f | gap=%.4f | decision=no_signal",
                tick.symbol,
                getattr(tick, "ts", None),
                getattr(tick, "price", None),
                p_long,
                p_short,
                signal_gap,
            )
            return None
        raw_order.meta = raw_order.meta or {}
        raw_order.meta.setdefault("p_long", p_long)
        raw_order.meta.setdefault("p_short", p_short)
        raw_order.meta.setdefault("signal_gap", signal_gap)
        raw_order.meta.setdefault("direction_prob", max(p_long, p_short))
        logger.info(
            "scan candidate | symbol=%s | ts=%s | price=%s | side=%s | p_long=%.4f | p_short=%.4f | gap=%.4f | score=%.4f",
            tick.symbol,
            getattr(tick, "ts", None),
            getattr(tick, "price", None),
            getattr(raw_order.side, "value", raw_order.side),
            p_long,
            p_short,
            signal_gap,
            float(raw_order.meta.get("score", 0.0)),
        )
        return _EntryCandidate(
            tick=tick,
            order=raw_order,
            prediction=prediction,
            features=df,
            score=float(raw_order.meta.get("score", 0.0)),
        )

    async def _command_for_entry_candidate(self, candidate: _EntryCandidate) -> TradingCommand | None:
        risk = self._deps["risk"]
        exchange = self._deps["exchange"]
        portfolio = self._deps["portfolio"]

        raw_order = candidate.order
        if isinstance(exchange, MarketOrderPreparationExchange):
            await exchange.prepare_market_order(raw_order)
        safe_order = risk.check(raw_order, portfolio, exchange)
        if safe_order is None:
            return None
        self._signal_seq += 1
        safe_order.meta = safe_order.meta or {}
        safe_order.meta["signal_number"] = self._signal_seq
        notification = await self._signal_notification(
            tick=candidate.tick,
            order=safe_order,
            portfolio=portfolio,
            features=candidate.features,
        )
        await self._notify_signal(notification)
        return PlaceOrderCommand(safe_order, reason="ENTRY", ts=getattr(candidate.tick, "ts", None), source_tick=candidate.tick)

    async def _commands_for_entry_tick(self, tick) -> list[TradingCommand]:
        candidate = await self._entry_candidate_for_tick(tick)
        if candidate is None:
            return []
        command = await self._command_for_entry_candidate(candidate)
        return [command] if command is not None else []

    async def _commands_for_market_event(self, event: MarketEvent) -> list[TradingCommand]:
        tick = event.tick
        self._prepare_tick_context(tick)
        exit_commands = await self._commands_for_exit_tick(tick, continue_with_entry=True)
        if exit_commands:
            return exit_commands
        return await self._commands_for_entry_tick(tick)

    async def process_event(self, event: TradingEvent) -> list[TradingCommand]:
        if isinstance(event, MarketEvent):
            return await self._commands_for_market_event(event)
        if isinstance(event, (OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent, FillEvent)):
            if not self._execution_dedupe.should_process(event):
                return []
        if isinstance(event, FillEvent):
            await self._apply_fill(event)
            if event.continue_with_entry and event.source_tick is not None:
                return await self._commands_for_entry_tick(event.source_tick)
            if event.command_reason == "ENTRY" and event.source_tick is not None:
                return await self._commands_for_exit_tick(event.source_tick, reason_suffix="INTRA-BAR")
            return []
        if isinstance(event, (OrderAcceptedEvent, OrderCancelledEvent, OrderRejectedEvent)):
            return []
        return []

    async def process_market_batch(self, ticks: list) -> list[TradingEvent]:
        """Process one timestamp batch fairly across symbols.

        Exits are executed before entries for every symbol in the batch. Entry
        candidates are then ranked by signal score before risk checks, so symbol
        iteration order does not decide which candidate gets limited capital.
        """
        if not ticks:
            return []

        ordered_ticks = sorted(ticks, key=lambda tick: str(getattr(tick, "symbol", "")))
        batch_ts = getattr(ordered_ticks[0], "ts", None)
        logger.info(
            "scan start | ts=%s | symbols=%s",
            batch_ts,
            ",".join(str(getattr(tick, "symbol", "")) for tick in ordered_ticks),
        )
        for tick in ordered_ticks:
            self._prepare_tick_context(tick)

        exit_commands: list[TradingCommand] = []
        for tick in ordered_ticks:
            exit_commands.extend(await self._commands_for_exit_tick(tick))
        events = await self.execute_commands(exit_commands)

        candidates: list[_EntryCandidate] = []
        for tick in ordered_ticks:
            candidate = await self._entry_candidate_for_tick(tick)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                float((candidate.order.meta or {}).get("direction_prob", 0.0)),
                float((candidate.order.meta or {}).get("signal_gap", 0.0)),
                str(candidate.order.symbol),
            ),
            reverse=True,
        )

        for candidate in candidates:
            command = await self._command_for_entry_candidate(candidate)
            if command is not None:
                events.extend(await self.execute_commands([command]))
        logger.info(
            "scan complete | ts=%s | symbols=%s | candidates=%s | events=%s",
            batch_ts,
            len(ordered_ticks),
            len(candidates),
            len(events),
        )
        return events

    async def execute_commands(self, commands: list[TradingCommand]) -> list[TradingEvent]:
        exchange = self._deps["exchange"]
        execution = self._deps["execution"]
        events: list[TradingEvent] = []
        pending = list(commands)
        while pending:
            command = pending.pop(0)
            if isinstance(command, (PlaceOrderCommand, CancelOrderCommand)):
                if isinstance(command, PlaceOrderCommand) and hasattr(execution, "ensure_client_order_id"):
                    execution.ensure_client_order_id(command.order, command)
                if self._event_journal is not None:
                    await self._event_journal.record_command(command, source="engine")
                command_events = await execution.execute(command, exchange)
                events.extend(command_events)
                for event in command_events:
                    if self._event_journal is not None:
                        await self._event_journal.record_event(event, source="execution")
                    pending.extend(await self.process_event(event))
        return events

    async def on_market_event(self, tick) -> None:
        await self.execute_commands(await self.process_event(MarketEvent(tick)))

    async def run(self, symbols: list[str]) -> None:
        self._running = True
        for symbol in symbols:
            self._deps["data"].subscribe(symbol, self.on_market_event)
        await self._deps["data"].run()
