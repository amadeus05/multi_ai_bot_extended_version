import logging
from dataclasses import dataclass
from pathlib import Path

from application.event_journal import EventJournal
from application.execution_event_deduplicator import ExecutionEventDeduplicator
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
from triple_barrier_chart import render_triple_barrier_chart


ENGINE_META_EXIT_REASON = "_engine_exit_reason"
ENGINE_META_REASON_SUFFIX = "_engine_reason_suffix"

logger = logging.getLogger(__name__)

SIGNAL_CHART_PATH = Path("charts") / "paper_signal_chart.png"


@dataclass(frozen=True)
class _EntryCandidate:
    tick: object
    order: Order
    prediction: dict
    features: object
    score: float


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


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

    def set_event_journal(self, journal: EventJournal | None) -> None:
        self._event_journal = journal

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

    @staticmethod
    def _order_position_side(order: Order) -> str:
        return "LONG" if order.side == OrderSide.BUY else "SHORT"

    @staticmethod
    def _barrier_price(entry_price: float, side: str, pct: float | None, *, is_take: bool) -> float | None:
        if pct is None:
            return None
        direction = 1.0 if side == "LONG" else -1.0
        sign = direction if is_take else -direction
        return float(entry_price) * (1.0 + sign * float(pct))

    async def _signal_chart_path(
        self,
        *,
        symbol: str,
        candles,
        entry_price: float,
        stop_price: float | None,
        take_price: float | None,
        probability: float,
    ) -> Path | None:
        if stop_price is None or take_price is None:
            return None

        try:
            if candles.empty:
                return None

            output_path = SIGNAL_CHART_PATH
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_triple_barrier_chart(
                candles,
                entry=entry_price,
                stop=stop_price,
                take=take_price,
                probability=probability,
                output_path=output_path,
                title=symbol,
            )
            return output_path
        except Exception:
            logger.exception("Signal chart generation failed | symbol=%s", symbol)
            return None

    async def _signal_notification(
        self,
        *,
        tick,
        order: Order,
        portfolio: PortfolioManager,
        features,
    ) -> SignalNotification:
        meta = order.meta or {}
        side = self._order_position_side(order)
        entry_price = float(order.price if order.price is not None else tick.price)
        stop_pct = _optional_float(meta.get("barrier_stop_pct"))
        take_pct = _optional_float(meta.get("barrier_take_pct"))
        stop_price = self._barrier_price(entry_price, side, stop_pct, is_take=False)
        take_price = self._barrier_price(entry_price, side, take_pct, is_take=True)
        direction_prob = float(meta.get("direction_prob", 0.0))
        chart_path = await self._signal_chart_path(
            symbol=order.symbol,
            candles=features,
            entry_price=entry_price,
            stop_price=stop_price,
            take_price=take_price,
            probability=direction_prob,
        )
        return SignalNotification(
            signal_id=int(meta.get("signal_number", self._signal_seq)),
            symbol=order.symbol,
            side=side,
            ts=getattr(tick, "ts", None),
            entry_price=entry_price,
            amount=float(order.amount),
            stop_price=stop_price,
            take_price=take_price,
            stop_pct=stop_pct,
            take_pct=take_pct,
            p_long=float(meta.get("p_long", 0.0)),
            p_short=float(meta.get("p_short", 0.0)),
            signal_gap=float(meta.get("signal_gap", 0.0)),
            direction_prob=direction_prob,
            proba_threshold=float(meta.get("directional_proba_threshold", 0.0)),
            min_signal_gap=float(meta.get("min_signal_gap", 0.0)),
            balance=float(portfolio.cash.get("USDT", 0.0)),
            chart_path=chart_path,
        )

    def _trade_exit_notification(self, res: dict, portfolio: PortfolioManager) -> TradeExitNotification:
        closed = portfolio.closed_trade_results
        wins = sum(1 for row in closed if float(row.get("pnl_abs", 0.0)) > 0)
        tp_count = sum(1 for row in closed if str(row.get("reason", "")).upper() == "TP")
        sl_count = sum(1 for row in closed if str(row.get("reason", "")).upper() == "SL")
        total = len(closed)
        balance = float(portfolio.cash.get("USDT", 0.0))
        pnl_abs = float(res.get("pnl_abs", 0.0))
        return TradeExitNotification(
            trade_number=int(res.get("trade_number", 0)),
            symbol=str(res.get("symbol", "")),
            side=str(res.get("side", "")).upper(),
            reason=str(res.get("reason", "CLOSE")),
            ts=res.get("ts"),
            entry_price=float(res.get("entry_price", 0.0)),
            exit_price=float(res.get("exit_price", 0.0)),
            qty=float(res.get("qty", 0.0)),
            pnl_abs=pnl_abs,
            pnl_pct=float(res.get("pnl_pct", 0.0)),
            commission=float(res.get("commission", 0.0)),
            balance=balance,
            entry_ts=res.get("entry_ts"),
            balance_before=balance - pnl_abs,
            winrate_pct=(wins / total * 100.0) if total else None,
            stop_losses_count=sl_count,
            take_profits_count=tp_count,
        )

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
