from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain.strategy.signal_resolver import resolve_directional_signal


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


if __name__ == "__main__":
    unittest.main()
