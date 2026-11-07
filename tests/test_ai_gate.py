import unittest

from ai_model import gate_status, GATE_MIN_WINS, GATE_MIN_LOSSES, FEATURE_DIM
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

    def test_incompatible_feature_snapshots_do_not_count(self):
        history = [self._signal(True, i) for i in range(30)]
        history += [self._signal(False, i) for i in range(30)]
        history[0]["feature_version"] = "old-schema"
        result = gate_status(history)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["compatible_signals"], 59)


if __name__ == "__main__":
    unittest.main()
