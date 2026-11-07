import unittest

from mutation_engine import _normalize, _mutate, evolve_population, FAMILIES


def sample_candles(count=300):
    rows = []
    price = 1.0
    for i in range(count):
        price *= 1.0004 if i % 9 else 0.9995
        rows.append({
            "ts": i, "o": price * 0.9998, "h": price * 1.001,
            "l": price * 0.999, "c": price, "vol": 1000 + (i % 13) * 100,
        })
    return rows


class MutationEngineTests(unittest.TestCase):
    def test_genome_is_bounded_and_reward_risk_is_valid(self):
        genome = _normalize({
            "family": "UNKNOWN", "tp_pct": 0.1, "sl_pct": 9,
            "trail_activation_pct": 100, "trail_gap_pct": 9,
        })
        self.assertIn(genome["family"], FAMILIES)
        self.assertGreaterEqual(genome["tp_pct"], 2.0)
        self.assertLessEqual(genome["tp_pct"], 6.0)
        self.assertGreaterEqual(genome["sl_pct"], 1.0)
        self.assertLessEqual(genome["sl_pct"], 3.0)
        self.assertGreaterEqual(genome["tp_pct"], genome["sl_pct"] * 1.5)
        self.assertLessEqual(genome["trail_activation_pct"], 70)
        self.assertLessEqual(genome["trail_gap_pct"], 1.5)

    def test_evolution_is_persistent_bounded_and_research_only(self):
        result = evolve_population(sample_candles(), {})
        self.assertEqual(result["report"]["population"], 16)
        self.assertEqual(result["report"]["generations_this_cycle"], 3)
        self.assertEqual(result["memory"]["generation"], 3)
        self.assertLessEqual(len(result["memory"]["archive"]), 24)
        self.assertFalse(result["report"]["live_execution_enabled"])
        next_result = evolve_population(sample_candles(), result["memory"])
        self.assertEqual(next_result["memory"]["generation"], 6)

    def test_insufficient_candles_does_not_mutate_memory(self):
        memory = {"generation": 7, "archive": []}
        result = evolve_population(sample_candles(80), memory)
        self.assertEqual(result["report"]["status"], "insufficient_candles")
        self.assertEqual(result["memory"]["generation"], 7)


if __name__ == "__main__":
    unittest.main()
