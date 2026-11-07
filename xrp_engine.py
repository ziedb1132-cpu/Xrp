"""XRP/USDT spot scanner with explicit market-data integrity checks; paper signals only."""
import math
import time
import requests

from feature_engine import build_features
from strategy_engine import select_strategy
from ai_model import select_entry_method

BASE = "https://www.okx.com/api/v5"


def _okx_data(response, label):
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code", "0")) != "0":
        raise RuntimeError(f"OKX {label} error: {payload.get('msg', 'unknown error')}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"OKX {label} returned no data")
    return data


def candles(bar="5m", limit=300):
    response = requests.get(
        BASE + "/market/candles",
        params={"instId": "XRP-USDT", "bar": bar, "limit": str(limit)},
        timeout=20,
    )
    rows = _okx_data(response, f"{bar} candles")
    parsed = []
    for row in reversed(rows):
        if len(row) < 6:
            raise RuntimeError(f"Malformed {bar} candle row")
        ts, o, h, low, close, volume = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        values = (o, h, low, close, volume)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite {bar} candle value")
        if min(o, h, low, close) <= 0 or volume < 0 or h < max(o, close, low) or low > min(o, close, h):
            raise RuntimeError(f"Invalid OHLCV values in {bar} candles")
        parsed.append({"ts": ts, "o": o, "h": h, "l": low, "c": close, "vol": volume})
    if len(parsed) < 50:
        raise RuntimeError(f"Insufficient {bar} candles: {len(parsed)}")
    timestamps = [item["ts"] for item in parsed]
    if timestamps != sorted(set(timestamps)):
        raise RuntimeError(f"Duplicate or out-of-order {bar} candle timestamps")
    return parsed


def current_price():
    response = requests.get(BASE + "/market/ticker", params={"instId": "XRP-USDT"}, timeout=20)
    ticker = _okx_data(response, "ticker")[0]
    price = float(ticker.get("last", 0))
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError("Invalid XRP-USDT ticker price")
    timestamp = int(ticker.get("ts", 0) or 0)
    if timestamp:
        age_ms = int(time.time() * 1000) - timestamp
        if age_ms < -30000 or age_ms > 120000:
            raise RuntimeError(f"Stale XRP-USDT ticker: age_ms={age_ms}")
    return price


def orderbook_context():
    try:
        response = requests.get(BASE + "/market/books", params={"instId": "XRP-USDT", "sz": "20"}, timeout=15)
        data = _okx_data(response, "order book")[0]
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            raise RuntimeError("Empty order book")
        bid = sum(float(item[1]) for item in bids)
        ask = sum(float(item[1]) for item in asks)
        bid_price = float(bids[0][0])
        ask_price = float(asks[0][0])
        mid = (bid_price + ask_price) / 2
        if min(bid, ask, bid_price, ask_price, mid) <= 0 or ask_price < bid_price:
            raise RuntimeError("Invalid order book levels")
        return {
            "imbalance": (bid - ask) / max(1e-12, bid + ask),
            "spread_pct": (ask_price - bid_price) / mid * 100,
            "available": True,
        }
    except Exception as exc:
        return {"imbalance": 0.0, "spread_pct": 0.0, "available": False, "error": str(exc)[:180]}


def _regime(cs, cs15, cs1h, cs4h):
    closes = [item["c"] for item in cs]
    returns = [closes[i] / closes[i - 1] - 1 for i in range(max(1, len(closes) - 20), len(closes)) if closes[i - 1]]
    trend = closes[-1] / closes[-21] - 1 if len(closes) >= 21 and closes[-21] else 0
    volatility = (sum(value * value for value in returns) / max(1, len(returns))) ** 0.5
    mtf = []
    for series in (cs15, cs1h, cs4h):
        values = [item["c"] for item in series]
        mtf.append(values[-1] / values[-5] - 1 if len(values) >= 5 and values[-5] else 0)
    regime = "bullish" if trend > 0.002 else ("bearish" if trend < -0.002 else "range")
    return {
        "regime": regime,
        "trend": round(trend, 6),
        "volatility": round(volatility, 6),
        "volatility_state": "high" if volatility > 0.006 else "normal",
        "trend_15m": round(mtf[0], 6),
        "trend_1h": round(mtf[1], 6),
        "trend_4h": round(mtf[2], 6),
    }


def analyze(news_sentiment=0.0, entry_method_stats=None, recent_stats=None):
    cs = candles("5m", 300)
    cs15 = candles("15m", 120)
    cs1h = candles("1H", 100)
    cs4h = candles("4H", 80)
    price = current_price()
    book = orderbook_context()
    context = _regime(cs, cs15, cs1h, cs4h)
    context.update({
        "book_imbalance": book.get("imbalance", 0.0),
        "book_spread_pct": book.get("spread_pct", 0.0),
        "orderbook_available": bool(book.get("available")),
        "candle_counts": {"5m": len(cs), "15m": len(cs15), "1h": len(cs1h), "4h": len(cs4h)},
        "latest_candle_ts": cs[-1]["ts"],
        "data_quality": "ok",
    })
    features = build_features(cs, context, book, news_sentiment)
    selected, ranked = select_strategy(cs, context, book, recent_stats)
    context["entry_diagnostics"] = {
        "selected": bool(selected),
        "selection_source": selected.get("selection_source") if selected else None,
        "opportunity_score": selected.get("opportunity_score") if selected else (ranked[0].get("opportunity_score") if ranked else None),
        "threshold": (ranked[0].get("opportunity_threshold") if ranked else None),
        "expected_net_edge_pct": (ranked[0].get("expected_net_edge_pct") if ranked else None),
        "confirmations": (ranked[0].get("confirmations") if ranked else 0),
        "reasons": (ranked[0].get("opportunity_reasons", []) if ranked else ["insufficient_candles"]),
        "best_strategy": ranked[0].get("name") if ranked else None,
        "active_legacy_strategies": sum(1 for item in ranked if item.get("active")),
        "reason": "qualified_best_opportunity" if selected else "no_setup_passed_independent_quality_gate",
    }
    entry_policy = select_entry_method(context, entry_method_stats)
    resistance = max(item["h"] for item in cs[-21:-1])
    support = min(item["l"] for item in cs[-21:-1])
    candidate = None
    if selected:
        candidate = {
            "side": "BUY",
            "entry": price,
            "tp": price * (1 + 1.30 / 100.0),
            "sl": price * (1 - 0.65 / 100.0),
            "tp_distance_pct": 1.30,
            "sl_distance_pct": 0.65,
            "strategy_name": selected["name"],
            "strategy_description": selected["description"],
            "strategy_score": selected["opportunity_score"],
            "opportunity_score": selected["opportunity_score"],
            "expected_net_edge_pct": ranked[0].get("expected_net_edge_pct"),
            "selection_source": selected.get("selection_source", "LEGACY_ACTIVE_STRATEGY"),
            "entry_reason": "Best available setup: " + selected["description"],
            "timeframe": "5m",
            "features": features,
            "feature_version": "xrp-1200-v2",
            "entry_method": entry_policy["method"],
            "entry_method_scores": entry_policy["scores"],
            "entry_method_samples": entry_policy["samples"],
            "entry_policy_source": entry_policy["source"],
            "entry_trigger_price": round(resistance * 1.0005, 10),
            "entry_retest_price": round(resistance, 10),
            "entry_support_price": round(support, 10),
            "entry_invalidation_price": round(support * 0.985, 10),
            "opportunity_score": selected.get("opportunity_score"),
        }
    return price, candidate, features, ranked, context


if __name__ == "__main__":
    from signal_manager import SignalManager
    current, setup, feature_set, strategies, market = analyze()
    SignalManager().cycle(current, setup, strategies, feature_set, market)
