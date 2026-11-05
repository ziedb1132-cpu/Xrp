"""Stable 1,200-dimensional XRP feature schema built from observed market data."""
import math

FEATURE_VERSION = "xrp-1200-v2"
BASE_COUNT = 200
TRANSFORMS = ("abs", "square", "signed_sqrt", "clip", "log1p_abs")

def _safe(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0

def build_features(candles, contexts=None, book=None, news_sentiment=0.0):
    if not candles:
        return {}
    closes = [_safe(x.get("c")) for x in candles]
    opens = [_safe(x.get("o")) for x in candles]
    highs = [_safe(x.get("h")) for x in candles]
    lows = [_safe(x.get("l")) for x in candles]
    vols = [_safe(x.get("vol")) for x in candles]
    last = closes[-1] or 1.0
    base = {}
    # 100 lagged returns, normalized to the latest close.
    for lag in range(1, 101):
        old = closes[-lag-1] if len(closes) > lag else last
        base[f"return_lag_{lag:03d}"] = (last / old - 1.0) if old else 0.0
    # 20 recent candles x 5 normalized candle/volume descriptors.
    for lag in range(20):
        idx = len(candles) - 1 - lag
        if idx < 0:
            o = h = l = c = v = last if lag == 0 else 0.0
        else:
            o, h, l, c, v = opens[idx], highs[idx], lows[idx], closes[idx], vols[idx]
        scale = c or last
        base[f"candle_{lag:02d}_body"] = (c-o)/scale
        base[f"candle_{lag:02d}_upper_wick"] = (h-max(o,c))/scale
        base[f"candle_{lag:02d}_lower_wick"] = (min(o,c)-l)/scale
        base[f"candle_{lag:02d}_range"] = (h-l)/scale
        prev_v = sum(vols[max(0,idx-19):idx+1])/max(1,len(vols[max(0,idx-19):idx+1])) if idx >= 0 else 0.0
        base[f"candle_{lag:02d}_relative_volume"] = v/prev_v-1 if prev_v else 0.0
    # Stable 200 raw + 1,000 transformed values. These are derived features,
    # not 1,200 independent market observations.
    out = {k: _safe(v) for k,v in base.items()}
    values = list(base.items())
    for transform in TRANSFORMS:
        for name, value in values:
            v = _safe(value)
            if transform == "abs": z = abs(v)
            elif transform == "square": z = v*v
            elif transform == "signed_sqrt": z = math.copysign(math.sqrt(abs(v)), v)
            elif transform == "clip": z = max(-1.0, min(1.0, v))
            else: z = math.copysign(math.log1p(abs(v)), v)
            out[f"{transform}__{name}"] = _safe(z)
    # Context features replace a fixed tail of reserved context keys.
    context = contexts or {}
    extras = {
        "context_news_sentiment": _safe(news_sentiment),
        "context_book_imbalance": _safe((book or {}).get("imbalance")),
        "context_book_spread_pct": _safe((book or {}).get("spread_pct")),
        "context_regime_trend": _safe(context.get("trend")),
        "context_regime_volatility": _safe(context.get("volatility")),
    }
    out.update(extras)
    # Keep a stable ordered schema at exactly 1,205 features.
    return dict(sorted(out.items()))
