import unittest

from scalping_engine import evaluate_scalping


def make_candles(count, step=0.00015):
    rows = []
    for i in range(count):
        close = 1.0 + i * step + 0.0003 * __import__("math").sin(i * 0.21)
        rows.append({
            "ts": i * 60000, "o": close - 0.0001, "h": close + 0.001,
            "l": close - 0.001, "c": close, "vol": 1500 if i == count - 1 else 1000,
            "confirmed": True,
        })
    return rows


class ScalpingEngineTests(unittest.TestCase):
    def test_valid_setup_returns_separate_fee_aware_scalp_profile(self):
        one = make_candles(300)
        five = make_candles(100, step=0.0005)
        candidate, diagnostics = evaluate_scalping(
            one, five, {"available": True, "spread_pct": 0.03, "imbalance": 0.12}, one[-1]["c"]
        )
        self.assertTrue(diagnostics["eligible"], diagnostics)
        self.assertEqual(candidate["strategy_type"], "SCALPING")
        self.assertGreaterEqual(candidate["tp_distance_pct"], 0.90)
        self.assertLessEqual(candidate["tp_distance_pct"], 1.50)
        self.assertGreaterEqual(candidate["sl_distance_pct"], 0.40)
        self.assertLessEqual(candidate["sl_distance_pct"], 0.80)
        self.assertGreaterEqual(candidate["tp_distance_pct"], candidate["sl_distance_pct"] * 1.5)
        self.assertEqual(candidate["entry_method"], "MARKET")

    def test_wide_spread_blocks_scalping(self):
        candidate, diagnostics = evaluate_scalping(
            make_candles(300), make_candles(100, step=0.0005),
            {"available": True, "spread_pct": 0.10, "imbalance": 0.2}, 1.05
        )
        self.assertIsNone(candidate)
        self.assertEqual(diagnostics["reason"], "spread_above_scalping_limit")

    def test_unconfirmed_latest_candle_is_not_used(self):
        one = make_candles(80)
        one[-1]["confirmed"] = False
        candidate, diagnostics = evaluate_scalping(
            one, make_candles(100, step=0.0005),
            {"available": True, "spread_pct": 0.02, "imbalance": 0.2}, 1.05
        )
        self.assertIsNone(candidate)
        self.assertEqual(diagnostics["reason"], "insufficient_market_history")


if __name__ == "__main__":
    unittest.main()
