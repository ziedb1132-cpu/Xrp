"""Opt-in live spot execution for XRP. Disabled unless XRP_LIVE_TRADING_ENABLED=true."""
import base64, hashlib, hmac, json, os, time
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode
import requests

BASE = "https://www.okx.com"
INST = "XRP-USDT"
STATE_FILE = "xrp_live_state.json"
TIMEOUT = 12

class OKXError(RuntimeError):
    pass

def _d(v):
    return Decimal(str(v))

def _state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"initialized": False, "baseline_signal_id": 0, "processed_signal_ids": [], "orders": [], "halted": False}

def _save(s):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, STATE_FILE)

class OKX:
    def __init__(self):
        self.key = os.getenv("OKX_API_KEY", "")
        self.secret = os.getenv("OKX_SECRET_KEY", "")
        self.passphrase = os.getenv("OKX_PASSPHRASE", "")
        if not all((self.key, self.secret, self.passphrase)):
            raise OKXError("Missing one or more OKX API secrets.")

    @staticmethod
    def _timestamp():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def request(self, method, path, body=None, private=True):
        body_text = json.dumps(body, separators=(",", ":")) if body is not None else ""
        ts = self._timestamp()
        headers = {"Content-Type": "application/json"}
        if private:
            prehash = ts + method.upper() + path + body_text
            sign = base64.b64encode(hmac.new(self.secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
            headers.update({"OK-ACCESS-KEY": self.key, "OK-ACCESS-SIGN": sign,
                            "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": self.passphrase})
        response = requests.request(method, BASE + path, headers=headers, data=body_text or None, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "0":
            raise OKXError("OKX API error: " + str(data.get("code")) + " " + str(data.get("msg")))
        return data.get("data", [])

    def available_usdt(self):
        rows = self.request("GET", "/api/v5/account/balance?ccy=USDT")
        for detail in (rows[0].get("details", []) if rows else []):
            if detail.get("ccy") == "USDT":
                return _d(detail.get("availBal") or detail.get("availEq") or "0")
        return Decimal("0")

    def instrument(self):
        rows = self.request("GET", "/api/v5/public/instruments?instType=SPOT&instId=" + INST, private=False)
        if not rows:
            raise OKXError("OKX did not return instrument metadata for " + INST)
        return rows[0]

    def get_order(self, cl_id):
        path = "/api/v5/trade/order?" + urlencode({"instId": INST, "clOrdId": cl_id})
        rows = self.request("GET", path)
        return rows[0] if rows else None

    def xrp_balance(self):
        rows = self.request("GET", "/api/v5/account/balance?ccy=XRP")
        detail = next((x for x in (rows[0].get("details", []) if rows else []) if x.get("ccy") == "XRP"), {})
        available = _d(detail.get("availBal") or "0")
        frozen = _d(detail.get("frozenBal") or "0")
        total = _d(detail.get("cashBal")) if detail.get("cashBal") not in (None, "") else available + frozen
        return {"available": available, "total": total}

    def cancel_order(self, cl_id, order_id=None):
        body = {"instId": INST, "ordId": str(order_id)} if order_id else {"instId": INST, "clOrdId": cl_id}
        rows = self.request("POST", "/api/v5/trade/cancel-order", body)
        if not rows or rows[0].get("sCode") != "0":
            raise OKXError("Could not cancel remaining XRP buy quantity: " + str(rows))
        return rows[0]

    def pending_algos(self):
        found = []
        for kind in ("oco", "conditional"):
            path = "/api/v5/trade/orders-algo-pending?" + urlencode({"ordType": kind, "instId": INST})
            found.extend(self.request("GET", path))
        return found

    def place_recovery_oco(self, signal, size, client_id):
        tp, sl, entry = _d(signal.get("tp")), _d(signal.get("sl")), _d(signal.get("entry"))
        if not (tp > entry > sl > 0):
            raise OKXError("Invalid TP/SL levels; refusing recovery OCO.")
        body = {"instId": INST, "tdMode": "cash", "side": "sell", "ordType": "oco",
                "sz": format(size, "f"), "algoClOrdId": client_id,
                "tpTriggerPx": format(tp, "f"), "tpOrdPx": "-1", "tpTriggerPxType": "last",
                "slTriggerPx": format(sl, "f"), "slOrdPx": "-1", "slTriggerPxType": "last"}
        rows = self.request("POST", "/api/v5/trade/order-algo", body)
        if not rows or rows[0].get("sCode") != "0" or not rows[0].get("algoId"):
            raise OKXError("OKX rejected recovery OCO: " + str(rows))
        return rows[0]

    def verify_protection(self, cl_id, record=None):
        """Confirm attached OCO or recover with a balance-delta-verified independent OCO."""
        order = self.get_order(cl_id)
        if not order:
            return {"verified": False, "state": "not_found"}
        state = str(order.get("state", "")).lower()
        filled = _d(order.get("accFillSz") or "0")
        if state in ("live", "partially_filled") and filled > 0:
            self.cancel_order(cl_id, order.get("ordId"))
            order = self.get_order(cl_id)
            if not order:
                return {"verified": False, "state": "cancel_reconciliation_pending"}
            state = str(order.get("state", "")).lower()
            filled = _d(order.get("accFillSz") or "0")
        if state in ("live", "partially_filled"):
            return {"verified": False, "state": "cancel_reconciliation_pending"}
        if state not in ("filled", "canceled", "mmp_canceled"):
            return {"verified": False, "state": state or "unknown"}
        if filled <= 0:
            return {"verified": False, "state": "canceled_no_fill"}
        expected = ("p" + cl_id[1:])[:32]
        algos = self.pending_algos()
        matches = [x for x in algos if x.get("algoClOrdId") == expected]
        if matches:
            algo = matches[0]
            tp_ok = _d(algo.get("tpTriggerPx")) > 0 and _d(algo.get("tpOrdPx")) == Decimal("-1")
            sl_ok = _d(algo.get("slTriggerPx")) > 0 and _d(algo.get("slOrdPx")) == Decimal("-1")
            if str(algo.get("ordType")) != "oco" or str(algo.get("side")) != "sell" or not (tp_ok and sl_ok):
                raise OKXError("URGENT: attached algo is not a valid sell OCO with both TP and SL.")
            return {"verified": True, "state": state, "algo_id": algo.get("algoId"),
                    "algo_client_id": expected, "tp_trigger": algo.get("tpTriggerPx"),
                    "sl_trigger": algo.get("slTriggerPx"), "algo_type": "oco", "size": algo.get("sz")}
        if record is None or "baseline_xrp_total" not in record or not isinstance(record.get("signal"), dict):
            raise OKXError("URGENT: cannot prove XRP ownership for recovery OCO; manual inspection required.")
        if algos:
            raise OKXError("URGENT: other XRP algo orders exist; refusing overlapping recovery sell.")
        pending_path = "/api/v5/trade/orders-pending?" + urlencode({"instType": "SPOT", "instId": INST})
        if self.request("GET", pending_path):
            raise OKXError("URGENT: other XRP spot orders exist; refusing overlapping recovery sell.")
        balance = self.xrp_balance()
        acquired = balance["total"] - _d(record["baseline_xrp_total"])
        meta = self.instrument()
        lot, minimum = _d(meta.get("lotSz", "0.0001")), _d(meta.get("minSz", "0"))
        tolerance = max(lot, filled * Decimal("0.01"))
        if acquired <= 0 or abs(filled - acquired) > tolerance:
            raise OKXError("URGENT: XRP balance delta does not match this order fill; cannot prove sell quantity.")
        size = (min(acquired, balance["available"]) / lot).to_integral_value(rounding=ROUND_DOWN) * lot
        if size < minimum:
            raise OKXError("URGENT: acquired XRP is not freely available or is below minimum order size.")
        recovery_id = ("r" + cl_id[1:])[:32]
        path = "/api/v5/trade/order-algo?" + urlencode({"algoClOrdId": recovery_id})
        existing = self.request("GET", path)
        if existing:
            algo = existing[0]
        else:
            placed = self.place_recovery_oco(record["signal"], size, recovery_id)
            algo_rows = self.request("GET", "/api/v5/trade/order-algo?" + urlencode({"algoId": placed["algoId"]}))
            algo = algo_rows[0] if algo_rows else {}
        tp_ok = _d(algo.get("tpTriggerPx")) > 0 and _d(algo.get("tpOrdPx")) == Decimal("-1")
        sl_ok = _d(algo.get("slTriggerPx")) > 0 and _d(algo.get("slOrdPx")) == Decimal("-1")
        if not algo or str(algo.get("ordType")) != "oco" or str(algo.get("side")) != "sell" or not (tp_ok and sl_ok):
            raise OKXError("URGENT: recovery OCO was not confirmed active by OKX.")
        return {"verified": True, "state": state, "algo_id": algo.get("algoId"),
                "algo_client_id": recovery_id, "tp_trigger": algo.get("tpTriggerPx"),
                "sl_trigger": algo.get("slTriggerPx"), "algo_type": "oco", "size": algo.get("sz")}

    def market_buy_with_protection(self, signal, quote_amount, cl_id):
        tp = _d(signal.get("tp", 0))
        sl = _d(signal.get("sl", 0))
        entry = _d(signal.get("entry", 0))
        if not (tp > entry > sl > 0):
            raise OKXError("Signal TP/SL levels are invalid; refusing live order.")
        meta = self.instrument()
        min_sz = _d(meta.get("minSz", "0"))
        lot_sz = _d(meta.get("lotSz", "0.0001"))
        estimated_base = quote_amount / entry
        rounded_base = (estimated_base / lot_sz).to_integral_value(rounding=ROUND_DOWN) * lot_sz
        if rounded_base < min_sz:
            raise OKXError("10% of available USDT is below the XRP minimum order size.")
        # Quote-currency market buy; TP and SL are attached to the parent order.
        body = {
            "instId": INST, "tdMode": "cash", "clOrdId": cl_id,
            "side": "buy", "ordType": "market", "sz": format(quote_amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN), "f"),
            "tgtCcy": "quote_ccy",
            "attachAlgoOrds": [{
                "attachAlgoClOrdId": ("p" + cl_id[1:])[:32],
                "tpTriggerPx": format(tp, "f"), "tpOrdPx": "-1", "tpTriggerPxType": "last",
                "slTriggerPx": format(sl, "f"), "slOrdPx": "-1", "slTriggerPxType": "last",
                "tpOrdKind": "condition"
            }]
        }
        rows = self.request("POST", "/api/v5/trade/order", body)
        if not rows or rows[0].get("sCode") != "0":
            raise OKXError("Order rejected: " + str(rows[0].get("sMsg") if rows else "empty response"))
        return rows[0]

def process_live(signal_state):
    """Process only newly created paper signals; never imports the pre-existing open signal."""
    if os.getenv("XRP_LIVE_TRADING_ENABLED", "false").strip().lower() != "true":
        return {"enabled": False, "status": "disabled"}
    if os.getenv("OKX_DEMO_TRADING", "false").strip().lower() == "true":
        return {"enabled": True, "status": "blocked", "reason": "Demo flag is not supported by this live adapter."}

    live = _state()
    current_total = int(signal_state.get("total_signals", 0))
    if not live.get("initialized"):
        live.update(initialized=True, baseline_signal_id=current_total)
        _save(live)
        return {"enabled": True, "status": "initialized", "baseline_signal_id": current_total}
    if live.get("halted"):
        return {"enabled": True, "status": "halted", "reason": live.get("halt_reason", "manual halt")}

    # Re-check accepted orders every cycle until OKX confirms active protection.
    pending = next((x for x in reversed(live.get("orders", [])) if x.get("status") == "protection_check_pending"), None)
    if pending:
        try:
            api = OKX()
            verification = api.verify_protection(pending["clOrdId"], pending)
            if verification.get("verified"):
                pending.update(status="protection_verified", verification=verification)
                live["last_status"] = "protection_verified"
                _save(live)
            else:
                pending.update(verification=verification)
                _save(live)
                return {"enabled": True, "status": "protection_check_pending", "signal_id": pending.get("signal_id"), "protection": verification}
        except Exception as exc:
            live["halted"] = True
            live["halt_reason"] = str(exc)[:500]
            pending.update(status="error_halted", error=str(exc)[:500])
            _save(live)
            return {"enabled": True, "status": "error_halted", "signal_id": pending.get("signal_id"), "reason": str(exc)[:300]}

    signal = signal_state.get("open_signal")
    if not isinstance(signal, dict) or signal.get("status") != "OPEN":
        return {"enabled": True, "status": "waiting"}
    sid = int(signal.get("id", 0))
    if sid <= int(live.get("baseline_signal_id", 0)):
        return {"enabled": True, "status": "skipped_existing_signal", "signal_id": sid}
    processed = set(int(x) for x in live.get("processed_signal_ids", []))
    if sid in processed:
        return {"enabled": True, "status": "already_processed", "signal_id": sid}

    # Persist the one-shot intent before any order request to prevent duplicate buys after a runner restart.
    live.setdefault("processed_signal_ids", []).append(sid)
    cl_id = "xrp" + str(sid).zfill(12)
    live.setdefault("orders", []).append({"signal_id": sid, "clOrdId": cl_id, "status": "request_started", "created_at": int(time.time())})
    _save(live)
    try:
        api = OKX()
        available = api.available_usdt()
        budget = (available * Decimal("0.10")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        record = live["orders"][-1]
        record["baseline_xrp_total"] = str(api.xrp_balance()["total"])
        record["signal"] = {k: signal.get(k) for k in ("entry", "tp", "sl")}
        _save(live)
        if budget <= 0:
            raise OKXError("No available USDT balance.")
        # Reconcile by client ID before sending; protects against a lost response/restart.
        existing = api.get_order(cl_id)
        if existing:
            result = existing
            status = "reconciled_existing_order"
        else:
            result = api.market_buy_with_protection(signal, budget, cl_id)
            status = "accepted"
        live["orders"][-1].update(status="protection_check_pending", order_id=result.get("ordId"), quote_budget=str(budget), response_state=result.get("state"))
        _save(live)
        verification = api.verify_protection(cl_id, live["orders"][-1])
        if verification.get("state") == "canceled_no_fill":
            live["orders"][-1].update(status="canceled_no_fill", verification=verification)
        else:
            live["orders"][-1].update(status="protection_verified" if verification["verified"] else "protection_check_pending", verification=verification, response_state=verification.get("state"))
        live["last_status"] = live["orders"][-1]["status"]
        _save(live)
        return {"enabled": True, "status": live["last_status"], "signal_id": sid, "order_id": result.get("ordId"), "budget_usdt": str(budget), "protection": verification}
    except Exception as exc:
        # Fail closed: this signal is not retried automatically; operator must inspect OKX and state.
        live["halted"] = True
        live["halt_reason"] = str(exc)[:500]
        live["orders"][-1].update(status="error_halted", error=str(exc)[:500])
        _save(live)
        return {"enabled": True, "status": "error_halted", "signal_id": sid, "reason": str(exc)[:300]}
