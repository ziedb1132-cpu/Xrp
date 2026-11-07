"""Conservative 1m/5m XRP scalping setup; fee-aware and isolated from swing rules."""
import math

BUY_FEE_PCT = 0.10
SELL_FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + SLIPPAGE_PCT


def _ema(values, period):
    if len(values) < period:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    value = float(values[0])
    for item in values[1:]:
        value = alpha * float(item) + (1.0 - alpha) * value
    return value


def _rsi(values, period=14):
    changes = [float(values[i]) - float(values[i - 1]) for i in range(max(1, len(values) - period), len(values))]
    gains = sum(max(0.0, x) for x in changes)
    losses = sum(max(0.0, -x) for x in changes)
    if losses <= 1e-12:
        return 100.0 if gains > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def _atr_pct(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    ranges = []
    for i in range(len(candles) - period, len(candles)):
        bar = candles[i]
        prev = float(candles[i - 1]["c"])
        ranges.append(max(float(bar["h"]) - float(bar["l"]),
                          abs(float(bar["h"]) - prev), abs(float(bar["l"]) - prev)))
    close = float(candles[-1]["c"])
    return sum(ranges) / period / close * 100.0 if close > 0 else 0.0


def evaluate_scalping(candles_1m, candles_5m, book, price):
    """Return a scalp candidate only when independent micro/macro and liquidity checks align."""
    # Never base a scalp decision on an unfinished exchange candle.
    candles_1m = candles_1m[:-1] if candles_1m and not candles_1m[-1].get("confirmed", True) else candles_1m
    candles_5m = candles_5m[:-1] if candles_5m and not candles_5m[-1].get("confirmed", True) else candles_5m
    if len(candles_1m) < 80 or len(candles_5m) < 40 or not isinstance(book, dict):
        return None, {"eligible": False, "reason": "insufficient_market_history"}
    if not book.get("available"):
        return None, {"eligible": False, "reason": "orderbook_unavailable"}
    spread = float(book.get("spread_pct", 999.0) or 999.0)
    imbalance = float(book.get("imbalance", 0.0) or 0.0)
    if not math.isfinite(spread) or spread > 0.06:
        return None, {"eligible": False, "reason": "spread_above_scalping_limit", "spread_pct": spread}

    c1 = [float(x["c"]) for x in candles_1m]
    c5 = [float(x["c"]) for x in candles_5m]
    if not all(math.isfinite(x) and x > 0 for x in c1[-40:] + c5[-40:]):
        return None, {"eligible": False, "reason": "invalid_candle_values"}

    ema9_1m, ema21_1m = _ema(c1[-40:], 9), _ema(c1[-40:], 21)
    ema9_5m, ema21_5m = _ema(c5[-40:], 9), _ema(c5[-40:], 21)
    rsi = _rsi(c1)
    momentum = c1[-1] / c1[-4] - 1.0 if c1[-4] else 0.0
    avg_volume = sum(float(x.get("vol", 0.0)) for x in candles_1m[-21:-1]) / 20.0
    volume_ratio = float(candles_1m[-1].get("vol", 0.0)) / avg_volume if avg_volume > 0 else 0.0
    atr = _atr_pct(candles_1m)

    checks = {
        "ema_1m_bullish": ema9_1m > ema21_1m and c1[-1] > ema9_1m,
        "ema_5m_confirmation": ema9_5m > ema21_5m and c5[-1] > ema9_5m,
        "momentum_positive": momentum >= 0.0005,
        "rsi_healthy": 48.0 <= rsi <= 68.0,
        "volume_confirmation": volume_ratio >= 1.15,
        "orderbook_not_selling": imbalance >= -0.05,
    }
    confirmations = sum(bool(x) for x in checks.values())
    # Reject dead markets and extreme one-minute volatility.
    if not 0.12 <= atr <= 0.75:
        return None, {"eligible": False, "reason": "atr_outside_scalping_band", "atr_pct": round(atr, 5), "checks": checks}
    if confirmations < 5:
        return None, {"eligible": False, "reason": "insufficient_scalping_confirmations",
                       "confirmations": confirmations, "checks": checks}

    # Dedicated scalp envelope: targets are deliberately above estimated round-trip costs.
    sl_pct = min(0.80, max(0.40, atr * 1.15))
    tp_pct = min(1.50, max(0.90, atr * 2.20, sl_pct * 1.60))
    if tp_pct < sl_pct * 1.50 or tp_pct <= ROUND_TRIP_COST_PCT * 2.0:
        return None, {"eligible": False, "reason": "reward_cost_ratio_failed", "tp_pct": tp_pct, "sl_pct": sl_pct}

    candidate = {
        "side": "BUY",
        "entry": float(price),
        "tp": float(price) * (1.0 + tp_pct / 100.0),
        "sl": float(price) * (1.0 - sl_pct / 100.0),
        "tp_distance_pct": round(tp_pct, 5),
        "sl_distance_pct": round(sl_pct, 5),
        "atr_pct": round(atr, 5),
        "strategy_type": "SCALPING",
        "strategy_name": "XRP_SCALPING_1M_5M",
        "strategy_description": "1m momentum and volume confirmed by 5m trend and tight order-book spread",
        "strategy_score": round(50.0 + confirmations * 7.0, 2),
        "opportunity_score": round(50.0 + confirmations * 7.0, 2),
        "expected_net_edge_pct": round(tp_pct - ROUND_TRIP_COST_PCT, 5),
        "selection_source": "SCALPING_MICROSTRUCTURE_FILTER",
        "entry_reason": "1m/5m momentum, volume, RSI and order-book filters aligned",
        "timeframe": "1m",
        "entry_method": "MARKET",
        "exit_distance_source": "SCALPING_ATR_FEE_AWARE",
        "trailing_activation_pct": 50.0,
        "trailing_stop_gap_pct": 0.25,
        "features": {},
        "feature_version": "xrp-scalping-1m-v1",
        "scalping_diagnostics": {
            "confirmations": confirmations, "checks": checks,
            "rsi_1m": round(rsi, 2), "momentum_3m_pct": round(momentum * 100.0, 5),
            "volume_ratio": round(volume_ratio, 3), "atr_pct": round(atr, 5),
            "spread_pct": round(spread, 5), "orderbook_imbalance": round(imbalance, 4),
            "estimated_round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        },
    }
    return candidate, {"eligible": True, **candidate["scalping_diagnostics"]}
