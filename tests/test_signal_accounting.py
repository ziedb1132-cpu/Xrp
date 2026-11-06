import json
import os
import tempfile
import unittest
from unittest.mock import patch

import signal_manager
from signal_manager import SignalManager, _close


class XRPPaperFeeAccountingTests(unittest.TestCase):
    def test_close_deducts_buy_fee_sell_fee_and_slippage(self):
        signal = {"entry": 100.0, "high_water": 102.0, "low_water": 99.0,
                  "strategy_name": "TEST", "entry_method": "MARKET"}
        state = {"history": [], "tp_count": 0, "sl_count": 0,
                 "strategy_stats": {}, "entry_method_stats": {}}
        with patch.object(signal_manager, "BUY_FEE_PCT", 0.15), \
             patch.object(signal_manager, "SELL_FEE_PCT", 0.08), \
             patch.object(signal_manager, "SLIPPAGE_PCT", 0.05):
            _close(signal, "TP", 101.5, 1234567890, state)

        self.assertAlmostEqual(signal["gross_return_pct"], 1.5, places=5)
        self.assertAlmostEqual(signal["buy_fee_pct"], 0.15, places=5)
        self.assertAlmostEqual(signal["sell_fee_pct"], 0.08, places=5)
        self.assertAlmostEqual(signal["fees_pct"], 0.23, places=5)
        self.assertAlmostEqual(signal["slippage_pct"], 0.05, places=5)
        self.assertAlmostEqual(signal["net_return_pct"], 1.22, places=5)
        self.assertEqual(signal["accounting_model"], "PAPER_ESTIMATE")
        self.assertEqual(len(state["history"]), 1)

    def test_summary_reports_gross_fees_slippage_and_net_separately(self):
        state = {
            "history": [{
                "status": "TP", "gross_return_pct": 1.5, "buy_fee_pct": 0.1,
                "sell_fee_pct": 0.1, "fees_pct": 0.2, "slippage_pct": 0.05,
                "net_return_pct": 1.25
            }],
            "total_signals": 1, "tp_count": 1, "sl_count": 0, "cycle": 1,
            "price": 1.25, "updated_at": 1234567890, "open_signal": None,
            "pending_signal": None, "entry_method_stats": {}, "ranked_strategies": [],
            "ai_gate": {}, "last_training": {}, "market_context": {}
        }
        with tempfile.TemporaryDirectory() as tmp, patch("os.getcwd", return_value=tmp):
            old = os.getcwd()
            try:
                os.chdir(tmp)
                SignalManager().write_summary(state)
                with open("xrp_summary.json", encoding="utf-8") as handle:
                    summary = json.load(handle)
            finally:
                os.chdir(old)

        self.assertEqual(summary["gross_profit_pct"], 1.5)
        self.assertEqual(summary["buy_fees_pct"], 0.1)
        self.assertEqual(summary["sell_fees_pct"], 0.1)
        self.assertEqual(summary["total_fees_pct"], 0.2)
        self.assertEqual(summary["slippage_pct"], 0.05)
        self.assertEqual(summary["net_profit_pct"], 1.25)
        self.assertEqual(summary["fee_model"], "PAPER_ESTIMATE")


if __name__ == "__main__":
    unittest.main()
