"""Explainable XRP opportunity scoring with fee-aware historical diagnostics."""
import math

STRATEGIES = [
    ("EMA_8_21_TREND", "Fast EMA above slow EMA with positive momentum"),
    ("EMA_13_34_TREND", "Medium EMA alignment with positive momentum"),
    ("MOMENTUM_6", "Positive six-candle momentum"),
    ("BREAKOUT_20", "Close breaks the previous 20-candle high"),
    ("RSI_RECOVERY", "RSI recovery from a weak zone"),
    ("VOLUME_BREAKOUT", "Positive candle with above-average volume"),
    ("VWAP_RECLAIM", "Close reclaims rolling volume-weighted price"),
    ("RANGE_EXPANSION", "Positive candle with expanding range"),
]
FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05
ROUND_TRIP_COST_PCT = 2 * FEE_PCT + SLIPPAGE_PCT


def _ema(values, period):
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    value = values[0]
    for x in values[1:]:
        value = x * k + value * (1 - k)
    return value


def _rsi(values, period=14):
    changes = [values[i] - values[i - 1] for i in range(max(1, len(values) - period), len(values))]
    gains = sum(max(0, x) for x in changes)
    losses = sum(max(0, -x) for x in changes)
    return 100.0 if losses == 0 and gains else (50.0 if losses == 0 else 100 - 100 / (1 + gains / losses))


def _flags(candles):
    closes = [float(x["c"]) for x in candles]
    volumes = [float(x.get("vol", 0)) for x in candles]
    if len(closes) < 40:
        return {}
    last = closes[-1]
    momentum = last / closes[-7] - 1 if closes[-7] else 0
    average_volume = sum(volumes[-21:-1]) / 20
    typical = [(float(x["h"]) + float(x["l"]) + float(x["c"])) / 3 for x in candles[-20:]]
    recent_volumes = volumes[-20:]
    vwap = sum(p * q for p, q in zip(typical, recent_volumes)) / max(1e-12, sum(recent_volumes))
    previous_range = float(candles[-2]["h"]) - float(candles[-2]["l"])
    last_range = float(candles[-1]["h"]) - float(candles[-1]["l"])
    return {
        "EMA_8_21_TREND": _ema(closes[-40:], 8) > _ema(closes[-40:], 21) and momentum > 0,
        "EMA_13_34_TREND": _ema(closes[-40:], 13) > _ema(closes[-40:], 34) and momentum > 0,
        "MOMENTUM_6": momentum > 0.002,
        "BREAKOUT_20": last > max(float(x["h"]) for x in candles[-21:-1]),
        "RSI_RECOVERY": 35 < _rsi(closes) < 58 and closes[-1] > closes[-2],
        "VOLUME_BREAKOUT": closes[-1] > closes[-2] and average_volume > 0 and volumes[-1] > 1.4 * average_volume,
        "VWAP_RECLAIM": closes[-2] <= vwap and last > vwap,
        "RANGE_EXPANSION": closes[-1] > closes[-2] and last_range > 1.25 * max(previous_range, 1e-12),
    }


def _opportunity(candles, context=None, book=None, recent_stats=None):
    context = context or {}
    book = book or {}
    closes = [float(x["c"]) for x in candles]
    volumes = [float(x.get("vol", 0)) for x in candles]
    if len(closes) < 40:
        return {"score": 0.0, "threshold": 100.0, "eligible": False, "reasons": ["insufficient_candles"]}
    last = closes[-1]
    ema8 = _ema(closes[-40:], 8)
    ema21 = _ema(closes[-40:], 21)
    rsi_now = _rsi(closes)
    rsi_prev = _rsi(closes[:-1])
    mom3 = last / closes[-4] - 1 if closes[-4] else 0.0
    mom6 = last / closes[-7] - 1 if closes[-7] else 0.0
    avg_volume = sum(volumes[-21:-1]) / 20
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0.0
    green = closes[-1] > closes[-2]
    score = 50.0
    reasons = []
    confirmations = 0

    if mom3 > 0.0008:
        score += 6; confirmations += 1; reasons.append("short_momentum_positive")
    if mom6 > 0.002:
        score += 7; confirmations += 1; reasons.append("six_candle_momentum_positive")
    if ema8 > ema21:
        score += 7; confirmations += 1; reasons.append("ema_alignment_bullish")
    if last > ema8:
        score += 4; confirmations += 1; reasons.append("price_above_fast_ema")
    if 42 <= rsi_now <= 68 and rsi_now >= rsi_prev:
        score += 7; confirmations += 1; reasons.append("rsi_recovery")
    if green:
        score += 4; confirmations += 1; reasons.append("positive_candle")
    if volume_ratio >= 1.10:
        score += 5; confirmations += 1; reasons.append("volume_confirmation")
    imbalance = float(book.get("imbalance", context.get("book_imbalance", 0)) or 0)
    spread = float(book.get("spread_pct", context.get("book_spread_pct", 0)) or 0)
    if imbalance >= 0.08:
        score += 4; confirmations += 1; reasons.append("buy_side_orderbook")
    elif imbalance <= -0.15:
        score -= 5; reasons.append("sell_side_orderbook")
    trends = [float(context.get(k, 0) or 0) for k in ("trend_15m", "trend_1h", "trend_4h")]
    aligned = sum(1 for x in trends if x > 0)
    opposed = sum(1 for x in trends if x < -0.001)
    score += aligned * 3
    score -= opposed * 3
    if aligned:
        reasons.append("higher_timeframe_alignment")
    regime = context.get("regime", "range")
    if regime == "bullish":
        score += 4
    elif regime == "bearish":
        score -= 8
        reasons.append("bearish_market_regime")
    volatility = float(context.get("volatility", 0) or 0)
    if volatility > 0.008:
        score -= 5
        reasons.append("elevated_volatility")
    if spread > 0.12:
        score -= 15
        reasons.append("spread_too_wide")

    threshold = 70.0
    closed = int((recent_stats or {}).get("closed", 0) or 0)
    win_rate = float((recent_stats or {}).get("win_rate_pct", 50) or 0)
    if closed >= 8 and win_rate < 35:
        threshold += 4
        reasons.append("recent_performance_risk_adjustment")
    if opposed >= 2:
        threshold += 3
    score = round(max(0.0, min(100.0, score)), 2)

    # Conservative, explicitly labelled estimate; this is a quality gate, not a calibrated probability.
    estimated_win = min(0.68, max(0.42, 0.50 + (score - 65.0) * 0.009))
    tp_pct, sl_pct = 1.30, 0.65
    net_win = tp_pct - ROUND_TRIP_COST_PCT
    net_loss = sl_pct + ROUND_TRIP_COST_PCT
    expected_net = estimated_win * net_win - (1 - estimated_win) * net_loss
    eligible = score >= threshold and confirmations >= 3 and spread <= 0.12 and expected_net > 0.03
    if confirmations < 3:
        reasons.append("too_few_independent_confirmations")
    if score < threshold:
        reasons.append("score_below_dynamic_threshold")
    if expected_net <= 0.03:
        reasons.append("insufficient_expected_net_edge")
    return {
        "score": score, "threshold": threshold, "confirmations": confirmations,
        "estimated_win_rate_pct": round(estimated_win * 100, 2),
        "expected_net_edge_pct": round(expected_net, 4),
        "tp_distance_pct": tp_pct, "sl_distance_pct": sl_pct,
        "eligible": eligible, "reasons": reasons,
        "momentum_3": round(mom3, 6), "momentum_6": round(mom6, 6),
        "rsi": round(rsi_now, 2), "volume_ratio": round(volume_ratio, 3),
        "orderbook_imbalance": round(imbalance, 4), "spread_pct": round(spread, 4),
    }


def _backtest(candles, name):
    trades = []
    fee = ROUND_TRIP_COST_PCT
    i = 40
    while i < len(candles) - 1:
        flags = _flags(candles[:i + 1])
        if not flags.get(name):
            i += 1
            continue
        entry = float(candles[i]["c"])
        tp, sl = entry * 1.012, entry * 0.993
        exit_price = None
        exit_index = None
        for j in range(i + 1, min(len(candles), i + 49)):
            bar = candles[j]
            if float(bar["l"]) <= sl:
                exit_price, exit_index = sl, j
                break
            if float(bar["h"]) >= tp:
                exit_price, exit_index = tp, j
                break
        if exit_price is None:
            i = min(len(candles) - 1, i + 48)
            continue
        trades.append((exit_price / entry - 1) * 100 - fee)
        i = exit_index + 1
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss else (99.0 if wins else 0.0)
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else None,
        "profit_factor": round(pf, 3),
        "expectancy_pct": round(sum(trades) / len(trades), 4) if trades else None,
        "eligible": len(trades) >= 20 and pf > 1.0 and sum(trades) > 0,
    }


def evaluate_strategies(candles, market_context=None, book=None, recent_stats=None):
    flags = _flags(candles)
    if not flags:
        return []
    opportunity = _opportunity(candles, market_context, book, recent_stats)
    results = []
    for name, description in STRATEGIES:
        active = bool(flags.get(name))
        bt = _backtest(candles, name)
        score = opportunity["score"] + (2.0 if active else 0.0) + (1.0 if bt["eligible"] else 0.0)
        results.append({
            "name": name, "description": description, "active": active,
            "score": round(min(100.0, score), 2),
            "opportunity_score": opportunity["score"],
            "opportunity_threshold": opportunity["threshold"],
            "confirmations": opportunity["confirmations"],
            "expected_net_edge_pct": opportunity["expected_net_edge_pct"],
            "opportunity_reasons": opportunity["reasons"],
            "eligible": opportunity["eligible"],
            "backtest_eligible": bt["eligible"],
            **bt,
        })
    return sorted(results, key=lambda x: (x["eligible"], x["opportunity_score"], x["active"], x["profit_factor"]), reverse=True)


def select_strategy(candles, market_context=None, book=None, recent_stats=None):
    ranked = evaluate_strategies(candles, market_context, book, recent_stats)
    if not ranked or not ranked[0].get("eligible"):
        return None, ranked
    # Best available technical setup can be selected even when none of the eight
    # legacy boolean triggers is active. This is deliberately not a forced entry.
    selected = dict(ranked[0])
    selected["selection_source"] = "BEST_AVAILABLE_OPPORTUNITY"
    return selected, ranked
