import unittest
from unittest.mock import patch

from okx_live import OKX, OKXError, process_live


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
    def test_live_execution_stays_disabled_without_opt_in(self):
        with patch.dict("os.environ", {"XRP_LIVE_TRADING_ENABLED": "false"}, clear=False):
            self.assertEqual(process_live({}), {"enabled": False, "status": "disabled"})

    def test_partial_fill_cancels_remainder_and_accepts_active_attached_oco(self):
        parent_partial = {"state": "partially_filled", "accFillSz": "3", "ordId": "parent-1"}
        parent_canceled = {"state": "canceled", "accFillSz": "3", "ordId": "parent-1"}
        attached = {"algoClOrdId": "p000000000001", "algoId": "algo-1", "ordType": "oco",
                    "side": "sell", "sz": "3", "tpTriggerPx": "1.3", "tpOrdPx": "-1",
                    "slTriggerPx": "1.1", "slOrdPx": "-1"}
        api = FakeOKX([parent_partial, parent_canceled], [attached])
        result = api.verify_protection("xrp000000000001", {})
        self.assertTrue(result["verified"])
        self.assertEqual(result["source"] if "source" in result else result["algo_type"], "oco")
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


if __name__ == "__main__":
    unittest.main()
