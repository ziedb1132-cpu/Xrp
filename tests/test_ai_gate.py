import unittest

from ai_model import gate_status, select_entry_method, _chronological_split, GATE_MIN_WINS, GATE_MIN_LOSSES, FEATURE_DIM
from feature_engine import FEATURE_VERSION


class BalancedAIGateTests(unittest.TestCase):
    def _signal(self, won, index):
        return {
            "status": "TP" if won else "SL",
            "net_return_pct": 0.5 if won else -0.5,
            "entry": 1.0,
            "high_water": 1.03 if won else 1.005,
            "low_water": 0.995 if won else 0.97,
            "feature_version": FEATURE_VERSION,
            "features": {f"feature_{i:04d}": 0.001 * (i + 1) for i in range(FEATURE_DIM)},
        }

    def test_gate_requires_30_wins_and_30_losses(self):
        history = [self._signal(True, i) for i in range(30)]
        history += [self._signal(False, i) for i in range(29)]
        self.assertFalse(gate_status(history)["enabled"])
        history.append(self._signal(False, 30))
        result = gate_status(history)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["wins"], GATE_MIN_WINS)
        self.assertEqual(result["losses"], GATE_MIN_LOSSES)

    def test_imbalanced_history_cannot_enable_ai(self):
        history = [self._signal(True, i) for i in range(59)]
        history.append(self._signal(False, 0))
        result = gate_status(history)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["wins"], 59)
        self.assertEqual(result["losses"], 1)

    def test_entry_method_learning_also_requires_balanced_outcomes(self):
        context = {"regime": "range", "trend": 0.0, "volatility": 0.001}
        losing = {"MARKET": {"trades": 60, "wins": 0, "losses": 60,
                             "avg_net_return_pct": -0.5, "win_rate_pct": 0.0}}
        result = select_entry_method(context, losing)
        self.assertFalse(result["learning_ready"])
        self.assertIn("30_WINS_30_LOSSES", result["source"])
        balanced = {"MARKET": {"trades": 60, "wins": 30, "losses": 30,
                               "avg_net_return_pct": 0.0, "win_rate_pct": 50.0}}
        self.assertTrue(select_entry_method(context, balanced)["learning_ready"])

    def test_chronological_validation_rejects_single_class_holdout(self):
        split, details = _chronological_split([0] * 80 + [1] * 20)
        self.assertIsNone(split)
        self.assertEqual(details["reason"], "chronological_split_lacks_both_classes")
        split, details = _chronological_split(([0, 1] * 50))
        self.assertIsNotNone(split)
        self.assertGreaterEqual(details["validation_wins"], 3)
        self.assertGreaterEqual(details["validation_losses"], 3)

    def test_incompatible_feature_snapshots_do_not_count(self):
        history = [self._signal(True, i) for i in range(30)]
        history += [self._signal(False, i) for i in range(30)]
        history[0]["feature_version"] = "old-schema"
        result = gate_status(history)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["compatible_signals"], 59)


if __name__ == "__main__":
    unittest.main()
