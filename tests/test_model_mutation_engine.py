import unittest

from model_mutation_engine import (
    FEATURE_FAMILIES, _normalize, evolve_model_inputs,
)


def candles(count=300):
    rows = []
    price = 0.5
    for i in range(count):
        change = 0.001 if i % 7 else -0.0004
        price *= 1 + change
        rows.append({
            "ts": i, "o": price * 0.9997, "h": price * 1.001,
            "l": price * 0.999, "c": price,
            "vol": 1000 + (i % 17) * 50, "confirmed": True,
        })
    return rows


class ModelMutationTests(unittest.TestCase):
    def test_feature_and_architecture_genes_are_bounded(self):
        g = _normalize({
            "features": ["returns", "fake", "rsi"],
            "transform": "unknown", "depth": 99, "width": 999,
            "dropout": 2, "learning_rate": 1, "weight_decay": -1,
        })
        self.assertEqual(g["features"], ["returns", "rsi"])
        self.assertEqual(g["transform"], "raw")
        self.assertEqual(g["depth"], 8)
        self.assertEqual(g["width"], 256)
        self.assertEqual(g["dropout"], 0.4)
        self.assertEqual(g["learning_rate"], 0.01)
        self.assertEqual(g["weight_decay"], 0.0)
        self.assertTrue(set(g["features"]).issubset(set(FEATURE_FAMILIES)))

    def test_evolution_reports_validation_and_untouched_test_without_deploying(self):
        result = evolve_model_inputs(candles(), {})
        report = result["report"]
        self.assertEqual(report["population"], 12)
        self.assertGreater(report["validation"]["samples"], 0)
        self.assertGreater(report["untouched_test"]["samples"], 0)
        self.assertFalse(report["live_schema_changed"])
        self.assertFalse(report["production_model_changed"])
        self.assertEqual(result["memory"]["generation"], 1)
        again = evolve_model_inputs(candles(), result["memory"])
        self.assertEqual(again["memory"]["generation"], 2)

    def test_short_history_does_not_start_evolution(self):
        memory = {"generation": 4, "archive": []}
        result = evolve_model_inputs(candles(80), memory)
        self.assertEqual(result["report"]["status"], "insufficient_candles")
        self.assertEqual(result["memory"]["generation"], 4)


if __name__ == "__main__":
    unittest.main()
