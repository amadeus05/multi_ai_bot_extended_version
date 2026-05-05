from application.trading_runtime_factory import build_trading_engine
from core.config.settings import TradingSettings
from domain.execution.execution_service import ExecutionService
from domain.portfolio.portfolio_manager import PortfolioManager


class FakeExchange:
    pass


class FakeDataProvider:
    pass


class FakeModel:
    def required_bars(self) -> int:
        return 1


def test_build_trading_engine_uses_supplied_portfolio_and_execution() -> None:
    portfolio = PortfolioManager(cash={"USDT": 123.0})
    execution = ExecutionService()

    engine = build_trading_engine(
        settings=TradingSettings(),
        exchange=FakeExchange(),
        data_provider=FakeDataProvider(),
        model=FakeModel(),
        portfolio=portfolio,
        execution=execution,
    )

    assert engine._deps["portfolio"] is portfolio
    assert engine._deps["execution"] is execution
