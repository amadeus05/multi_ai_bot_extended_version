from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalKline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float


@dataclass(frozen=True)
class FundingRatePoint:
    funding_time: int
    funding_rate: float


@dataclass(frozen=True)
class OpenInterestPoint:
    timestamp: int
    open_interest: float
