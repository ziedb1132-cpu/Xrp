import json
import os
import tempfile
import unittest
from unittest.mock import patch

from okx_live import OKX, OKXError, process_live, _quote_budget, _scalping_paper_gate


class FakeOKX(OKX):
    def __init__(self, orders, algos=None, acquired="3"):
        self.orders = list(orders)
        self.algos = list(algos or [])
        self.cancelled = []
        self.placed = []
        self.acquired = acquired

    def get_order(self, cl_id):
        return self.orders.pop(0) if self.orders else None

    def cancel_order(self, cl_id, order_id=None):
        self.cancelled.append((cl_id, order_id))
        return {"sCode": "0"}

    def pending_algos(self):
        return self.algos

    def xrp_balance(self):
        from decimal import Decimal
        return {"available": Decimal(self.acquired), "total": Decimal(self.acquired)}

    def instrument(self):
        return {"lotSz": "0.1", "minSz": "0.1"}

    def request(self, method, path, body=None, private=True):
        if "/orders-pending?" in path:
            return []
        if "algoClOrdId=" in path:
            return []
        if "algoId=" in path:
            return [{"algoId": "algo-1", "state": "live", "ordType": "oco", "side": "sell",
                     "sz": "3", "tpTriggerPx": "1.3", "tpOrdPx": "-1",
                     "slTriggerPx": "1.1", "slOrdPx": "-1"}]
        raise AssertionError("Unexpected API request in offline test: " + path)

    def place_recovery_oco(self, signal, size, client_id):
        self.placed.append((signal, str(size), client_id))
        return {"algoId": "algo-1", "sCode": "0"}


class XRPExecutionSafetyTests(unittest.TestCase):
    def test_quote_budget_uses_100_percent_of_available_usdt(self):
        from decimal import Decimal
        self.assertEqual(_quote_budget(Decimal("123.456789")), Decimal("123.45"))
        self.assertEqual(_quote_budget(Decimal("0.009")), Decimal("0.00"))

    def test_live_execution_stays_disabled_without_opt_in(self):
        with patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "false"}, clear=False):
            self.assertEqual(process_live({}), {"enabled": False, "status": "disabled"})

    def test_disabled_cycle_is_persisted_to_live_state_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "xrp_live_state.json")
            with patch("okx_live.STATE_FILE", state_path):
                with patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "false"}, clear=True):
                    result = process_live({"open_signal": {"id": 12, "status": "OPEN"}})
                self.assertEqual(result, {"enabled": False, "status": "disabled"})
                with open(state_path, encoding="utf-8") as handle:
                    saved = json.load(handle)
                self.assertEqual(saved["last_attempt"]["status"], "disabled")
                self.assertEqual(saved["last_attempt"]["signal_id"], 12)
                self.assertFalse(saved["last_attempt"]["enabled"])
                self.assertEqual(len(saved["attempts"]), 1)

    def test_partial_fill_cancels_remainder_and_accepts_active_attached_oco(self):
        parent_partial = {"state": "partially_filled", "accFillSz": "3", "ordId": "parent-1"}
        parent_canceled = {"state": "canceled", "accFillSz": "3", "ordId": "parent-1"}
        attached = {"algoClOrdId": "p000000000001", "algoId": "algo-1", "ordType": "oco",
                    "side": "sell", "sz": "3", "tpTriggerPx": "1.3", "tpOrdPx": "-1",
                    "slTriggerPx": "1.1", "slOrdPx": "-1"}
        api = FakeOKX([parent_partial, parent_canceled], [attached])
        result = api.verify_protection("xrp000000000001", {})
        self.assertTrue(result["verified"])
        self.assertEqual(result["algo_type"], "oco")
        self.assertEqual(len(api.cancelled), 1)
        self.assertEqual(api.placed, [])

    def test_missing_attached_protection_installs_and_verifies_recovery_oco(self):
        parent = {"state": "filled", "accFillSz": "3", "ordId": "parent-2"}
        api = FakeOKX([parent], [], acquired="3")
        rec = {"clOrdId": "xrp000000000002", "baseline_xrp_total": "0",
               "signal": {"entry": "1.2", "tp": "1.3", "sl": "1.1"}}
        result = api.verify_protection(rec["clOrdId"], rec)
        self.assertTrue(result["verified"])
        self.assertEqual(result["algo_type"], "oco")
        self.assertEqual(len(api.placed), 1)
        self.assertEqual(api.placed[0][1], "3")

    def test_balance_mismatch_halts_instead_of_creating_a_sell(self):
        parent = {"state": "filled", "accFillSz": "3", "ordId": "parent-3"}
        api = FakeOKX([parent], [], acquired="1")
        rec = {"clOrdId": "xrp000000000003", "baseline_xrp_total": "0",
               "signal": {"entry": "1.2", "tp": "1.3", "sl": "1.1"}}
        with self.assertRaises(OKXError):
            api.verify_protection(rec["clOrdId"], rec)
        self.assertEqual(api.placed, [])

    def test_wrong_oco_size_is_rejected(self):
        parent = {"state": "filled", "accFillSz": "3", "ordId": "parent-4"}
        wrong = {"algoClOrdId": "p000000000004", "algoId": "algo-4", "ordType": "oco",
                 "side": "sell", "sz": "1", "tpTriggerPx": "1.3", "tpOrdPx": "-1",
                 "slTriggerPx": "1.1", "slOrdPx": "-1"}
        api = FakeOKX([parent], [wrong])
        with self.assertRaises(OKXError):
            api.verify_protection("xrp000000000004", {})
        self.assertEqual(api.placed, [])



class XRPAutomaticResumeTests(unittest.TestCase):
    def _saved_state(self, path, reason="No available USDT balance."):
        state = {
            "initialized": True, "baseline_signal_id": 4, "processed_signal_ids": [5],
            "orders": [{"signal_id": 5, "clOrdId": "xrp000000000005",
                        "status": "error_halted", "error": reason}],
            "halted": True, "halt_reason": reason, "attempts": []
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)

    def test_positive_balance_resumes_without_replaying_old_signal(self):
        from decimal import Decimal
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xrp_live_state.json")
            self._saved_state(path)
            api = Mock()
            api.available_usdt.return_value = Decimal("25.00")
            api.request.return_value = []
            api.pending_algos.return_value = []
            api.xrp_balance.return_value = {"available": Decimal("0"), "total": Decimal("0")}
            with patch("okx_live.STATE_FILE", path), patch("okx_live.OKX", return_value=api), patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "true", "OKX_DEMO_TRADING": "false"}, clear=False):
                result = process_live({"total_signals": 13, "open_signal": None})
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(result["status"], "resumed_waiting_for_new_signal")
            self.assertFalse(saved["halted"])
            self.assertEqual(saved["processed_signal_ids"], [5])
            self.assertEqual(saved["orders"][0]["status"], "skipped_no_funds")
            self.assertEqual(saved["resume_check"]["status"], "resumed")

    def test_zero_balance_keeps_halt_active(self):
        from decimal import Decimal
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xrp_live_state.json")
            self._saved_state(path)
            api = Mock()
            api.available_usdt.return_value = Decimal("0")
            with patch("okx_live.STATE_FILE", path), patch("okx_live.OKX", return_value=api), patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "true", "OKX_DEMO_TRADING": "false"}, clear=False):
                result = process_live({"total_signals": 13, "open_signal": None})
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(result["status"], "waiting_for_usdt")
            self.assertTrue(saved["halted"])
            api.request.assert_not_called()

    def test_existing_xrp_or_orders_prevent_automatic_resume(self):
        from decimal import Decimal
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xrp_live_state.json")
            self._saved_state(path)
            api = Mock()
            api.available_usdt.return_value = Decimal("25")
            api.request.return_value = [{"ordId": "manual-open-order"}]
            api.pending_algos.return_value = []
            api.xrp_balance.return_value = {"available": Decimal("0"), "total": Decimal("0")}
            with patch("okx_live.STATE_FILE", path), patch("okx_live.OKX", return_value=api), patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "true", "OKX_DEMO_TRADING": "false"}, clear=False):
                result = process_live({"total_signals": 13, "open_signal": None})
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(result["status"], "halted_reconciliation_required")
            self.assertTrue(saved["halted"])

    def test_non_balance_safety_halt_is_never_auto_cleared(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xrp_live_state.json")
            self._saved_state(path, "URGENT: unknown XRP position.")
            with patch("okx_live.STATE_FILE", path), patch("okx_live.OKX") as api_class, patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "true", "OKX_DEMO_TRADING": "false"}, clear=False):
                result = process_live({"total_signals": 13, "open_signal": None})
            self.assertEqual(result["status"], "halted")
            api_class.assert_not_called()



class XRPScalpingLiveGateTests(unittest.TestCase):
    def test_scalping_live_requires_thirty_profitable_paper_closes(self):
        history = (
            [{"strategy_type": "SCALPING", "status": "TP", "net_return_pct": 1.0} for _ in range(20)]
            + [{"strategy_type": "SCALPING", "status": "SL", "net_return_pct": -0.5} for _ in range(10)]
        )
        gate = _scalping_paper_gate({"history": history})
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["closed_scalps"], 30)
        self.assertGreater(gate["profit_factor"], 1.05)
        self.assertGreater(gate["net_profit_pct"], 0)

    def test_scalping_live_stays_blocked_with_insufficient_or_unprofitable_history(self):
        short_history = [
            {"strategy_type": "SCALPING", "status": "TP", "net_return_pct": 1.0}
            for _ in range(29)
        ]
        self.assertFalse(_scalping_paper_gate({"history": short_history})["eligible"])
        losing_history = (
            [{"strategy_type": "SCALPING", "status": "TP", "net_return_pct": 0.2} for _ in range(10)]
            + [{"strategy_type": "SCALPING", "status": "SL", "net_return_pct": -0.5} for _ in range(20)]
        )
        self.assertFalse(_scalping_paper_gate({"history": losing_history})["eligible"])

if __name__ == "__main__":
    unittest.main()
