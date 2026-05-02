import asyncio
import inspect

from core.interfaces.data_provider import DataProvider
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
from core.types.order_flags import ORDER_META_FILL_PRICE_FINAL
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager
from domain.risk.risk_manager import RiskManager
from domain.ml.labeling.barrier_policy import BarrierPolicy
from domain.ml.labeling.models import LabelingConfig
from domain.strategy.strategy_engine import Strategy


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
        execution_listener=None,
        barrier_policy: BarrierPolicy | None = None,
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
        self._pending: set[asyncio.Task] = set()
        self._execution_listener = execution_listener

    def set_execution_listener(self, listener) -> None:
        self._execution_listener = listener

    @staticmethod
    def _tick_price(tick, name: str, fallback: float) -> float:
        value = getattr(tick, name, None)
        if value is None:
            return float(fallback)
        return float(value)

    async def _notify_execution(self, trade, portfolio) -> None:
        if self._execution_listener is None:
            return
        maybe_coro = self._execution_listener(trade, portfolio)
        if inspect.isawaitable(maybe_coro):
            await maybe_coro

    async def _process_exit_for_tick(self, tick, *, reason_suffix: str | None = None) -> bool:
        risk = self._deps["risk"]
        portfolio = self._deps["portfolio"]
        exchange = self._deps["exchange"]
        execution = self._deps["execution"]
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
                        meta={ORDER_META_FILL_PRICE_FINAL: True},
                    )
                    exit_trade = await execution.execute(exit_order, exchange)
                    exit_trade.ts = getattr(tick, "ts", exit_trade.ts)
                    
                    prev_closed = len(portfolio.closed_trade_results)
                    portfolio.apply_execution(exit_trade)
                    
                    for res in portfolio.closed_trade_results[prev_closed:]:
                        res["reason"] = reason
                        is_sl = (reason == "SL")
                        if hasattr(risk, "register_trade_result"):
                            risk.register_trade_result(res["symbol"], res["pnl_abs"], exit_trade.ts, stop_loss_hit=is_sl)
                    if portfolio.trade_events and portfolio.trade_events[-1].get("type") == "CLOSE":
                        portfolio.trade_events[-1]["reason"] = reason
                        if reason_suffix:
                            portfolio.trade_events[-1]["reason_suffix"] = reason_suffix
                    await self._notify_execution(exit_trade, portfolio)
                    return True
        return False

    async def _handle_tick(self, tick) -> None:
        risk = self._deps["risk"]
        exchange = self._deps["exchange"]
        portfolio = self._deps["portfolio"]
        execution = self._deps["execution"]

        if hasattr(exchange, "set_last_price"):
            exchange.set_last_price(float(tick.price))

        if hasattr(risk, "set_current_bar"):
            risk.set_current_bar(getattr(tick, "ts", None))

        await self._process_exit_for_tick(tick)
        if portfolio.get_position(tick.symbol) is not None:
            return

        df = await self._deps["data"].warmup(tick.symbol, self._deps["model"].required_bars())
        prediction = self._deps["model"].predict(df)
        bp = self._barrier_policy.barriers_for_last_row(df)
        if bp is not None:
            prediction["barrier_stop_pct"], prediction["barrier_take_pct"] = bp
        raw_order = self._deps["strategy"].on_prediction(tick, prediction, self._deps["portfolio"])
        if raw_order is None:
            return
        raw_order.meta = raw_order.meta or {}
        p_long = float(prediction.get("p_long", 0.5))
        p_short = float(prediction.get("p_short", 0.5))
        raw_order.meta.update(
            {
                "p_long": p_long,
                "p_short": p_short,
                "signal_gap": abs(p_long - p_short),
                "direction_prob": max(p_long, p_short),
            }
        )
        safe_order = risk.check(raw_order, portfolio, exchange)
        if safe_order is None:
            return
        trade = await execution.execute(safe_order, exchange)
        trade.ts = getattr(tick, "ts", trade.ts)
        
        # Save barriers into trade/position
        if safe_order.meta:
            trade.meta = trade.meta or {}
            trade.meta.update(safe_order.meta)
            
        portfolio.apply_execution(trade)
        await self._notify_execution(trade, portfolio)
        await self._process_exit_for_tick(tick, reason_suffix="INTRA-BAR")

    async def on_market_event(self, tick) -> None:
        await self._handle_tick(tick)

    async def close_all_at_market(self, market_prices: dict[str, float], ts) -> None:
        from core.types.domain_types import Order
        from core.types.enums import OrderSide

        portfolio = self._deps["portfolio"]
        exchange = self._deps["exchange"]
        execution = self._deps["execution"]

        for pos in list(portfolio.get_open_positions()):
            ref_price = float(market_prices.get(pos.symbol, pos.entry_price))
            side = OrderSide.SELL if pos.side.value == "long" else OrderSide.BUY
            if hasattr(exchange, "market_fill_price"):
                exit_price = exchange.market_fill_price(ref_price, side)
            else:
                exit_price = ref_price
            order = Order(
                symbol=pos.symbol,
                side=side,
                amount=pos.amount,
                price=exit_price,
                meta={ORDER_META_FILL_PRICE_FINAL: True, "reason": "FINAL"},
            )
            trade = await execution.execute(order, exchange)
            trade.ts = ts
            prev_closed = len(portfolio.closed_trade_results)
            portfolio.apply_execution(trade)
            for res in portfolio.closed_trade_results[prev_closed:]:
                res["reason"] = "FINAL"
            if portfolio.trade_events and portfolio.trade_events[-1].get("type") == "CLOSE":
                portfolio.trade_events[-1]["reason"] = "FINAL"
            await self._notify_execution(trade, portfolio)

    def _track_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def run(self, symbols: list[str]) -> None:
        self._running = True
        for symbol in symbols:
            self._deps["data"].subscribe(symbol, lambda tick: self._track_task(self._handle_tick(tick)))
        await self._deps["data"].run()
        if self._pending:
            results = await asyncio.gather(*self._pending, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result
