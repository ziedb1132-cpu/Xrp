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

    def verify_protection(self, cl_id):
        """Verify filled parent order and its actually active OCO/conditional protection."""
        order = self.get_order(cl_id)
        if not order:
            return {"verified": False, "state": "not_found"}
        state = str(order.get("state", "")).lower()
        filled = _d(order.get("accFillSz") or "0")
        if state == "partially_filled" or (filled > 0 and state != "filled"):
            raise OKXError("URGENT: XRP order is partially filled. Live protection is not verified; trading halted. Inspect OKX and manage the filled XRP manually.")
        if state != "filled":
            return {"verified": False, "state": state or "unknown"}
        expected_algo_id = ("p" + cl_id[1:])[:32]
        found = []
        for algo_type in ("conditional", "oco"):
            path = "/api/v5/trade/orders-algo-pending?" + urlencode({"ordType": algo_type, "instId": INST})
            found.extend(self.request("GET", path))
        matches = [x for x in found if x.get("algoClOrdId") == expected_algo_id]
        if not matches:
            raise OKXError("URGENT: XRP buy filled but no active attached TP/SL algo found at OKX. Trading halted; inspect account immediately.")
        algo = matches[0]
        tp_ok = _d(algo.get("tpTriggerPx") or "0") > 0 and bool(algo.get("tpOrdPx"))
        sl_ok = _d(algo.get("slTriggerPx") or "0") > 0 and bool(algo.get("slOrdPx"))
        if not (tp_ok and sl_ok):
            raise OKXError("URGENT: active OKX algo does not confirm both TP and SL triggers. Trading halted; inspect account immediately.")
        return {"verified": True, "state": state, "algo_id": algo.get("algoId"),
                "algo_client_id": expected_algo_id, "tp_trigger": algo.get("tpTriggerPx"),
                "sl_trigger": algo.get("slTriggerPx"), "algo_type": algo.get("ordType")}

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
        verification = api.verify_protection(cl_id)
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
