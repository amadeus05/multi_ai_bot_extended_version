import asyncio
import inspect

from core.interfaces.data_provider import DataProvider
from core.interfaces.exchange import Exchange
from core.interfaces.model import Model
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

    async def _handle_tick(self, tick) -> None:
        risk = self._deps["risk"]
        portfolio = self._deps["portfolio"]
        exchange = self._deps["exchange"]
        execution = self._deps["execution"]
        
        if hasattr(risk, "set_current_bar"):
            risk.set_current_bar(getattr(tick, "ts", None))
            
        # 1. Process exits first (Phase 4)
        if self.exit_manager is not None:
            pos = portfolio.get_position(tick.symbol)
            if pos is not None:
                # In live, next_open/high/low are all just the current tick price
                exit_price, reason = self.exit_manager.check_causal_exit(
                    position=pos,
                    next_open=tick.price,
                    next_high=tick.price,
                    next_low=tick.price,
                    stop_pct=pos.meta.get("barrier_stop_pct", float('inf')) if pos.meta else float('inf'),
                    take_pct=pos.meta.get("barrier_take_pct", float('inf')) if pos.meta else float('inf')
                )
                if exit_price is not None:
                    from core.types.enums import OrderSide
                    exit_side = OrderSide.SELL if pos.side.value == "LONG" else OrderSide.BUY
                    from core.types.domain_types import Order
                    from infrastructure.exchanges.simulation.simulated_exchange import (
                        ORDER_META_SIM_FILL_PRICE_FINAL,
                    )
                    exit_order = Order(
                        symbol=pos.symbol,
                        side=exit_side,
                        amount=pos.amount,
                        price=exit_price,
                        meta={ORDER_META_SIM_FILL_PRICE_FINAL: True},
                    )
                    exit_trade = await execution.execute(exit_order, exchange)
                    exit_trade.ts = getattr(tick, "ts", exit_trade.ts)
                    
                    prev_closed = len(portfolio.closed_trade_results)
                    portfolio.apply_execution(exit_trade)
                    
                    # 2. Register risk outcome (Phase 2)
                    for res in portfolio.closed_trade_results[prev_closed:]:
                        res["reason"] = reason
                        is_sl = (reason == "SL")
                        if hasattr(risk, "register_trade_result"):
                            risk.register_trade_result(res["symbol"], res["pnl_abs"], exit_trade.ts, stop_loss_hit=is_sl)
                    
                    if self._execution_listener is not None:
                        maybe_coro = self._execution_listener(exit_trade, portfolio)
                        import inspect
                        if inspect.isawaitable(maybe_coro):
                            await maybe_coro

        df = await self._deps["data"].warmup(tick.symbol, self._deps["model"].required_bars())
        prediction = self._deps["model"].predict(df)
        bp = self._barrier_policy.barriers_for_last_row(df)
        if bp is not None:
            prediction["barrier_stop_pct"], prediction["barrier_take_pct"] = bp
        raw_order = self._deps["strategy"].on_prediction(tick, prediction, self._deps["portfolio"])
        if raw_order is None:
            return
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
        if self._execution_listener is not None:
            maybe_coro = self._execution_listener(trade, portfolio)
            import inspect
            if inspect.isawaitable(maybe_coro):
                await maybe_coro

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
