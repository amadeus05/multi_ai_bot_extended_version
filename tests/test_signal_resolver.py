from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.types.domain_types import Tick
from core.types.enums import OrderSide
from domain.strategy.score_threshold_strategy import BacktestParitySignalStrategy
from domain.strategy.signal_resolver import build_entry_score, resolve_directional_signal


class ResolveDirectionalSignalTest(unittest.TestCase):
    def test_exact_tie_is_neutral(self) -> None:
        signal, probability, gap = resolve_directional_signal(0.5, 0.5, 0.5, 0.0)

        self.assertEqual(signal, 0)
        self.assertEqual(probability, 0.5)
        self.assertEqual(gap, 0.0)

    def test_weak_edge_below_gap_is_neutral(self) -> None:
        signal, probability, gap = resolve_directional_signal(0.52, 0.48, 0.5, 0.05)

        self.assertEqual(signal, 0)
        self.assertEqual(probability, 0.52)
        self.assertAlmostEqual(gap, 0.04)

    def test_confident_long_passes_threshold_and_gap(self) -> None:
        signal, probability, gap = resolve_directional_signal(0.58, 0.42, 0.55, 0.05)

        self.assertEqual(signal, 1)
        self.assertEqual(probability, 0.58)
        self.assertAlmostEqual(gap, 0.16)

    def test_confident_short_passes_threshold_and_gap(self) -> None:
        signal, probability, gap = resolve_directional_signal(0.41, 0.59, 0.55, 0.05)

        self.assertEqual(signal, -1)
        self.assertEqual(probability, 0.59)
        self.assertAlmostEqual(gap, 0.18)

    def test_build_entry_score_matches_backtest_formula(self) -> None:
        self.assertAlmostEqual(build_entry_score(0.58, 0.16, 0.55), 0.46)

    def test_strategy_uses_shared_entry_score(self) -> None:
        strategy = BacktestParitySignalStrategy(
            directional_proba_threshold=0.55,
            min_signal_gap=0.05,
            allow_longs=True,
            allow_shorts=True,
        )
        tick = Tick(
            symbol="BTC/USDT",
            ts=pd.Timestamp("2024-01-01T00:00:00"),
            bid=100.0,
            ask=100.0,
            price=100.0,
            volume=1.0,
        )

        order = strategy.on_prediction(
            tick,
            {"p_long": 0.58, "p_short": 0.42, "barrier_stop_pct": 0.02, "barrier_take_pct": 0.04},
            portfolio=None,
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertAlmostEqual(order.meta["score"], build_entry_score(0.58, 0.16, 0.55))
        self.assertAlmostEqual(order.meta["direction_prob"], 0.58)
        self.assertAlmostEqual(order.meta["signal_gap"], 0.16)


if __name__ == "__main__":
    unittest.main()
