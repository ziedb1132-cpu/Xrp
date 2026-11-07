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

def _quote_budget(available):
    """Allocate 100% of currently available USDT, rounded down to OKX quote precision."""
    return _d(available).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

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
        expected = ("p" + cl_id[3:])[:32]
        algos = self.pending_algos()
        matches = [x for x in algos if x.get("algoClOrdId") == expected]
        if matches:
            algo = matches[0]
            tp_ok = _d(algo.get("tpTriggerPx")) > 0 and _d(algo.get("tpOrdPx")) == Decimal("-1")
            sl_ok = _d(algo.get("slTriggerPx")) > 0 and _d(algo.get("slOrdPx")) == Decimal("-1")
            algo_size = _d(algo.get("sz") or "0")
            meta = self.instrument()
            lot = _d(meta.get("lotSz", "0.0001"))
            if (str(algo.get("ordType")) != "oco" or str(algo.get("side")) != "sell"
                    or not (tp_ok and sl_ok) or abs(algo_size - filled) > max(lot, filled * Decimal("0.01"))):
                raise OKXError("URGENT: attached algo is not a correctly sized sell OCO with both TP and SL.")
            return {"verified": True, "state": state, "algo_id": algo.get("algoId"),
                    "algo_client_id": expected, "tp_trigger": algo.get("tpTriggerPx"),
                    "sl_trigger": algo.get("slTriggerPx"), "algo_type": "oco", "size": algo.get("sz")}
        # An attached OCO may already have triggered and disappeared from the pending list.
        attached_path = "/api/v5/trade/order-algo?" + urlencode({"instId": INST, "algoClOrdId": expected})
        attached_history = self.request("GET", attached_path)
        if attached_history:
            attached = attached_history[0]
            tp_ok = _d(attached.get("tpTriggerPx") or "0") > 0 and _d(attached.get("tpOrdPx") or "0") == Decimal("-1")
            sl_ok = _d(attached.get("slTriggerPx") or "0") > 0 and _d(attached.get("slOrdPx") or "0") == Decimal("-1")
            attached_size = _d(attached.get("sz") or "0")
            meta = self.instrument()
            lot = _d(meta.get("lotSz", "0.0001"))
            if (str(attached.get("state", "")).lower() == "live" and str(attached.get("ordType")) == "oco"
                    and str(attached.get("side")) == "sell" and tp_ok and sl_ok
                    and abs(attached_size - filled) <= max(lot, filled * Decimal("0.01"))):
                return {"verified": True, "state": state, "algo_id": attached.get("algoId"),
                        "algo_client_id": expected, "tp_trigger": attached.get("tpTriggerPx"),
                        "sl_trigger": attached.get("slTriggerPx"), "algo_type": "oco", "size": attached.get("sz")}
            raise OKXError("URGENT: attached OCO is no longer active; manual exit reconciliation is required.")
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
        recovery_id = ("r" + cl_id[3:])[:32]
        path = "/api/v5/trade/order-algo?" + urlencode({"instId": INST, "algoClOrdId": recovery_id})
        existing = self.request("GET", path)
        if existing:
            algo = existing[0]
        else:
            placed = self.place_recovery_oco(record["signal"], size, recovery_id)
            algo_rows = self.request("GET", "/api/v5/trade/order-algo?" + urlencode({"instId": INST, "algoId": placed["algoId"]}))
            algo = algo_rows[0] if algo_rows else {}
        tp_ok = _d(algo.get("tpTriggerPx")) > 0 and _d(algo.get("tpOrdPx")) == Decimal("-1")
        sl_ok = _d(algo.get("slTriggerPx")) > 0 and _d(algo.get("slOrdPx")) == Decimal("-1")
        if (not algo or str(algo.get("state", "")).lower() != "live"
                or str(algo.get("ordType")) != "oco" or str(algo.get("side")) != "sell"
                or not (tp_ok and sl_ok)):
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
            raise OKXError("100% of available USDT is below the XRP minimum order size.")
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

def _process_live(signal_state):
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
        # Only the specific zero-balance halt is eligible for automatic recovery.
        # Other safety halts always require manual reconciliation.
        halt_reason = str(live.get("halt_reason", "manual halt"))
        if halt_reason != "No available USDT balance.":
            return {"enabled": True, "status": "halted", "reason": halt_reason}
        try:
            api = OKX()
            available = api.available_usdt()
            if available <= 0:
                return {"enabled": True, "status": "waiting_for_usdt", "available_usdt": str(available)}
            # Do not resume into an unknown position/order left outside our saved state.
            pending_orders = api.request(
                "GET", "/api/v5/trade/orders-pending?" + urlencode({"instType": "SPOT", "instId": INST})
            )
            pending_algos = api.pending_algos()
            balance = api.xrp_balance()
            if pending_orders or pending_algos or balance["total"] > 0:
                reason = "USDT is available, but existing XRP balance or open XRP orders require manual reconciliation."
                live["halt_reason"] = reason
                live["resume_check"] = {
                    "timestamp": int(time.time()), "available_usdt": str(available),
                    "pending_spot_orders": len(pending_orders), "pending_algo_orders": len(pending_algos),
                    "xrp_total": str(balance["total"]), "status": "manual_reconciliation_required"
                }
                _save(live)
                return {"enabled": True, "status": "halted_reconciliation_required", "reason": reason}
            # The original request stopped before POST /trade/order because budget was zero.
            # Keep its signal ID processed: never replay an old signal after a later deposit.
            for order in reversed(live.get("orders", [])):
                if order.get("status") == "error_halted" and order.get("error") == "No available USDT balance." and not order.get("order_id"):
                    order["status"] = "skipped_no_funds"
                    order["reconciled_at"] = int(time.time())
                    break
            live["halted"] = False
            live["halt_reason"] = ""
            live["last_status"] = "resumed_waiting_for_new_signal"
            live["resume_check"] = {
                "timestamp": int(time.time()), "available_usdt": str(available),
                "pending_spot_orders": 0, "pending_algo_orders": 0,
                "xrp_total": str(balance["total"]), "status": "resumed"
            }
            _save(live)
            return {"enabled": True, "status": "resumed_waiting_for_new_signal",
                    "available_usdt": str(available), "allocation_pct": 100}
        except Exception as exc:
            # A failed verification must never clear the halt.
            return {"enabled": True, "status": "halted", "reason": str(exc)[:300]}

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

    # Never stack a new live buy while an earlier protective OCO is still active.
    previous = next((x for x in reversed(live.get("orders", [])) if x.get("status") == "protection_verified"), None)
    if previous:
        try:
            api = OKX()
            algo_id = (previous.get("verification") or {}).get("algo_id")
            if not algo_id:
                raise OKXError("Previous XRP position has no recorded protective OCO ID.")
            if any(x.get("algoId") == algo_id for x in api.pending_algos()):
                return {"enabled": True, "status": "position_open", "algo_id": algo_id}
            raise OKXError("Previous XRP OCO is no longer pending. Verify its sell fill and remaining XRP manually before allowing another buy.")
        except Exception as exc:
            live["halted"] = True
            live["halt_reason"] = str(exc)[:500]
            _save(live)
            return {"enabled": True, "status": "error_halted", "reason": str(exc)[:300]}

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
        budget = _quote_budget(available)
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
        live["orders"][-1].update(status="protection_check_pending", order_id=result.get("ordId"), quote_budget=str(budget), allocation_pct=100, response_state=result.get("state"))
        _save(live)
        verification = api.verify_protection(cl_id, live["orders"][-1])
        if verification.get("state") == "canceled_no_fill":
            live["orders"][-1].update(status="canceled_no_fill", verification=verification)
        else:
            live["orders"][-1].update(status="protection_verified" if verification["verified"] else "protection_check_pending", verification=verification, response_state=verification.get("state"))
        live["last_status"] = live["orders"][-1]["status"]
        _save(live)
        return {"enabled": True, "status": live["last_status"], "signal_id": sid, "order_id": result.get("ordId"), "budget_usdt": str(budget), "allocation_pct": 100, "protection": verification}
    except Exception as exc:
        # Fail closed: this signal is not retried automatically; operator must inspect OKX and state.
        live["halted"] = True
        live["halt_reason"] = str(exc)[:500]
        live["orders"][-1].update(status="error_halted", error=str(exc)[:500])
        _save(live)
        return {"enabled": True, "status": "error_halted", "signal_id": sid, "reason": str(exc)[:300]}


def process_live(signal_state):
    """Run one live-execution check and persist its outcome for dashboard/audit use."""
    started_at = int(time.time())
    signal = signal_state.get("open_signal") if isinstance(signal_state, dict) else None
    signal_id = signal.get("id") if isinstance(signal, dict) else None
    result = None
    error = None
    try:
        result = _process_live(signal_state)
        return result
    except Exception as exc:
        error = str(exc)[:500]
        raise
    finally:
        # Audit every cycle, including disabled/waiting/initialized outcomes. This is
        # deliberately separate from order records and never triggers another request.
        live = _state()
        outcome = result if isinstance(result, dict) else {}
        attempt = {
            "timestamp": started_at,
            "enabled": os.getenv("XRP_LIVE_TRADING_ENABLED", "false").strip().lower() == "true",
            "signal_id": outcome.get("signal_id", signal_id),
            "status": outcome.get("status", "exception" if error else "unknown"),
        }
        for key in ("order_id", "algo_id", "budget_usdt", "reason", "protection"):
            if key in outcome:
                attempt[key] = outcome[key]
        if error:
            attempt["error"] = error
        attempts = live.setdefault("attempts", [])
        attempts.append(attempt)
        # Keep the audit file bounded while retaining recent cycle-by-cycle outcomes.
        live["attempts"] = attempts[-500:]
        live["last_attempt"] = attempt
        _save(live)
