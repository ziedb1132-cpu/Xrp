import unittest
from unittest.mock import patch

from strategy_engine import _backtest, _opportunity, select_strategy


def rising_candles(count=300):
    rows = []
    for i in range(count):
        price = 1.0 + i * 0.0005
        rows.append({
            "ts": i, "o": price - 0.0001, "h": price + 0.0002,
            "l": price - 0.0002, "c": price, "vol": 1000 + (i % 5) * 20,
        })
    return rows


class OpportunityEngineTests(unittest.TestCase):
    def test_strong_setup_passes_independent_quality_gate(self):
        candles = rising_candles()
        context = {
            "regime": "bullish", "trend_15m": 0.004,
            "trend_1h": 0.006, "trend_4h": 0.003, "volatility": 0.002,
        }
        book = {"imbalance": 0.20, "spread_pct": 0.02}
        result = _opportunity(candles, context, book)
        self.assertTrue(result["eligible"], result)
        self.assertGreaterEqual(result["confirmations"], 3)
        self.assertGreater(result["expected_net_edge_pct"], 0.03)

    def test_exit_distances_are_adaptive_and_risk_bounded(self):
        candles = rising_candles()
        result = _opportunity(
            candles,
            {"regime": "bullish", "trend_15m": 0.004, "trend_1h": 0.006, "trend_4h": 0.003},
            {"imbalance": 0.20, "spread_pct": 0.02},
        )
        self.assertGreaterEqual(result["tp_distance_pct"], 2.0)
        self.assertGreaterEqual(result["sl_distance_pct"], 1.0)
        self.assertLessEqual(result["tp_distance_pct"], 6.0)
        self.assertLessEqual(result["sl_distance_pct"], 3.0)
        self.assertGreaterEqual(result["tp_distance_pct"], result["sl_distance_pct"] * 1.5)
        self.assertEqual(result["exit_distance_source"], "ATR_VOLATILITY_ADAPTIVE")

    def test_wide_spread_blocks_entry(self):
        result = _opportunity(
            rising_candles(),
            {"regime": "bullish", "trend_15m": 0.004, "trend_1h": 0.006, "trend_4h": 0.003},
            {"imbalance": 0.20, "spread_pct": 0.20},
        )
        self.assertFalse(result["eligible"])
        self.assertIn("spread_too_wide", result["reasons"])

    def test_unavailable_orderbook_blocks_entry(self):
        result = _opportunity(
            rising_candles(),
            {"regime": "bullish", "trend_15m": 0.004, "trend_1h": 0.006, "trend_4h": 0.003},
            {"imbalance": 0.0, "spread_pct": 999.0, "available": False},
        )
        self.assertFalse(result["eligible"])
        self.assertIn("orderbook_unavailable", result["reasons"])
        self.assertIn("spread_too_wide", result["reasons"])

    def test_backtest_keeps_unhit_trade_open_instead_of_inventing_timeout(self):
        candles = [{
            "ts": i, "o": 100.0, "h": 100.2, "l": 99.8, "c": 100.0, "vol": 1000.0
        } for i in range(80)]
        with patch("strategy_engine._flags", return_value={"TEST": True}):
            result = _backtest(candles, "TEST")
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["unresolved_at_end"], 1)
        self.assertNotIn("time_exits", result)

    def test_best_opportunity_does_not_require_legacy_boolean_trigger(self):
        candles = rising_candles()
        with patch("strategy_engine._flags", return_value={}), patch(
            "strategy_engine._backtest",
            return_value={"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": None,
                          "profit_factor": 0.0, "expectancy_pct": None, "eligible": False},
        ):
            selected, ranked = select_strategy(
                candles,
                {"regime": "bullish", "trend_15m": 0.004, "trend_1h": 0.006, "trend_4h": 0.003},
                {"imbalance": 0.20, "spread_pct": 0.02},
            )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["selection_source"], "BEST_AVAILABLE_OPPORTUNITY")
        self.assertEqual(len(ranked), 8)


if __name__ == "__main__":
    unittest.main()
