"""Research-only evolution of XRP input features and model hyperparameters.

This module never changes the production feature schema, model checkpoint or order flow.
It evaluates feature recipes on chronological train/validation/test partitions.
"""
import math
import random

FEATURE_FAMILIES = (
    "returns", "volatility", "candle_shape", "volume", "rsi", "ema_gap"
)
TRANSFORMS = ("raw", "abs", "signed_sqrt", "clip")
HORIZONS = (3, 6, 12)


def _clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _normalize(g):
    g = dict(g or {})
    families = [x for x in g.get("features", []) if x in FEATURE_FAMILIES]
    families = list(dict.fromkeys(families))[:len(FEATURE_FAMILIES)]
    if not families:
        families = ["returns", "volatility", "volume"]
    transform = g.get("transform", "raw")
    if transform not in TRANSFORMS:
        transform = "raw"
    depth = _clamp_int(g.get("depth", 3), 2, 8)
    width = _clamp_int(g.get("width", 64), 32, 256)
    return {
        "features": families,
        "transform": transform,
        "horizon": int(g.get("horizon", 6)) if int(g.get("horizon", 6)) in HORIZONS else 6,
        "depth": depth,
        "width": width,
        "dropout": round(max(0.0, min(0.4, float(g.get("dropout", 0.1)))), 3),
        "learning_rate": round(max(0.00001, min(0.01, float(g.get("learning_rate", 0.001)))), 6),
        "weight_decay": round(max(0.0, min(0.01, float(g.get("weight_decay", 0.0001)))), 6),
    }


def _features(candles, i, genome):
    rows = candles[:i + 1]
    closes = [float(x["c"]) for x in rows]
    opens = [float(x["o"]) for x in rows]
    highs = [float(x["h"]) for x in rows]
    lows = [float(x["l"]) for x in rows]
    vols = [float(x.get("vol", 0)) for x in rows]
    c = closes[-1]
    out = []
    for family in genome["features"]:
        if family == "returns":
            for p in (2, 3, 6, 12, 24):
                out.append(c / closes[-p-1] - 1 if len(closes) > p and closes[-p-1] else 0.0)
        elif family == "volatility":
            for p in (6, 12, 24):
                rr = [closes[k] / closes[k-1] - 1 for k in range(max(1, len(closes)-p), len(closes)) if closes[k-1]]
                out.append(math.sqrt(sum(x*x for x in rr) / max(1, len(rr))))
        elif family == "candle_shape":
            scale = c or 1.0
            out.extend(((c-opens[-1])/scale, (highs[-1]-max(c, opens[-1]))/scale,
                        (min(c, opens[-1])-lows[-1])/scale, (highs[-1]-lows[-1])/scale))
        elif family == "volume":
            avg = sum(vols[-21:-1]) / max(1, len(vols[-21:-1]))
            out.append(vols[-1] / avg - 1 if avg else 0.0)
        elif family == "rsi":
            for p in (7, 14, 21):
                changes = [closes[k]-closes[k-1] for k in range(max(1, len(closes)-p), len(closes))]
                gains = sum(max(0, x) for x in changes)
                losses = sum(max(0, -x) for x in changes)
                rsi = 100.0 if losses == 0 and gains else (50.0 if losses == 0 else 100-100/(1+gains/losses))
                out.append((rsi-50)/50)
        elif family == "ema_gap":
            for fast, slow in ((5, 13), (8, 21), (13, 34)):
                def ema(vals, period):
                    k = 2/(period+1)
                    value = vals[0]
                    for v in vals[1:]:
                        value = v*k + value*(1-k)
                    return value
                window = closes[-max(40, slow*3):]
                out.append((ema(window, fast)/ema(window, slow)-1) if ema(window, slow) else 0.0)
    transformed = []
    for value in out:
        value = value if math.isfinite(value) else 0.0
        if genome["transform"] == "abs":
            value = abs(value)
        elif genome["transform"] == "signed_sqrt":
            value = math.copysign(math.sqrt(abs(value)), value)
        elif genome["transform"] == "clip":
            value = max(-1.0, min(1.0, value))
        transformed.append(value)
    return transformed


def _corr(a, b):
    if len(a) < 8:
        return 0.0
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)
    vb = sum((x-mb)**2 for x in b)
    if va <= 1e-18 or vb <= 1e-18:
        return 0.0
    return sum((x-ma)*(y-mb) for x,y in zip(a,b)) / math.sqrt(va*vb)


def _evaluate(candles, genome, start, end):
    horizon = genome["horizon"]
    matrix, targets = [], []
    for i in range(max(40, start), min(end, len(candles)-horizon)):
        if not all(candles[k].get("confirmed", True) for k in (i, i+horizon)):
            continue
        entry, future = float(candles[i]["c"]), float(candles[i+horizon]["c"])
        if entry <= 0:
            continue
        matrix.append(_features(candles, i, genome))
        targets.append(future/entry-1)
    if len(targets) < 8:
        return {"samples": len(targets), "mean_abs_correlation": 0.0, "directional_accuracy": 0.0}
    correlations = [_corr([row[j] for row in matrix], targets) for j in range(len(matrix[0]))]
    abs_corr = [abs(x) for x in correlations]
    prediction = [sum(row[j] * correlations[j] for j in range(len(correlations))) for row in matrix]
    correct = sum(1 for p, y in zip(prediction, targets) if (p >= 0) == (y >= 0))
    return {
        "samples": len(targets),
        "mean_abs_correlation": round(sum(abs_corr)/max(1,len(abs_corr)), 6),
        "directional_accuracy": round(correct/len(targets), 5),
        "feature_correlations": [round(x, 5) for x in correlations],
    }


def _random(rng):
    return _normalize({
        "features": rng.sample(list(FEATURE_FAMILIES), rng.randint(2, len(FEATURE_FAMILIES))),
        "transform": rng.choice(TRANSFORMS),
        "horizon": rng.choice(HORIZONS),
        "depth": rng.randint(2, 8), "width": rng.choice((32, 64, 96, 128, 192, 256)),
        "dropout": rng.uniform(0.02, 0.3),
        "learning_rate": 10 ** rng.uniform(-4.5, -2.0),
        "weight_decay": 10 ** rng.uniform(-6, -2),
    })


def _mutate(parent, rng):
    g = dict(parent)
    if rng.random() < 0.6:
        family = rng.choice(FEATURE_FAMILIES)
        features = list(g["features"])
        if family in features and len(features) > 2:
            features.remove(family)
        elif family not in features:
            features.append(family)
        g["features"] = features
    if rng.random() < 0.35: g["transform"] = rng.choice(TRANSFORMS)
    if rng.random() < 0.3: g["horizon"] = rng.choice(HORIZONS)
    if rng.random() < 0.5: g["depth"] += rng.choice((-2, -1, 1, 2))
    if rng.random() < 0.5: g["width"] = rng.choice((32, 64, 96, 128, 192, 256))
    if rng.random() < 0.5: g["dropout"] += rng.gauss(0, 0.04)
    if rng.random() < 0.5: g["learning_rate"] *= math.exp(rng.gauss(0, 0.5))
    if rng.random() < 0.5: g["weight_decay"] *= math.exp(rng.gauss(0, 0.7))
    return _normalize(g)


def evolve_model_inputs(candles, memory=None):
    """Evolve input recipes and suggest architecture/hyperparameters; research only."""
    memory = dict(memory or {})
    if len(candles) < 100:
        return {"memory": memory, "report": {"status": "insufficient_candles", "promoted": False}}
    rng = random.Random(int(candles[-1].get("ts", len(candles))) ^ int(memory.get("generation", 0)) ^ 9173)
    archive = [_normalize(x) for x in memory.get("archive", []) if isinstance(x, dict)]
    population = archive[:8]
    while len(population) < 12:
        population.append(_random(rng))
    n = len(candles)
    train_end = int(n*0.60)
    validation_end = int(n*0.80)
    scored = []
    for genome in population:
        train = _evaluate(candles, genome, 40, train_end)
        validation = _evaluate(candles, genome, train_end, validation_end)
        # Selection pressure is deliberately based on development segment only.
        fitness = train["mean_abs_correlation"] * max(0.0, train["directional_accuracy"]-0.5)
        scored.append({"genome": genome, "train": train, "validation": validation, "fitness": fitness})
    scored.sort(key=lambda x: x["fitness"], reverse=True)
    parents = [x["genome"] for x in scored[:4]]
    next_population = parents[:]
    while len(next_population) < 12:
        next_population.append(_mutate(rng.choice(parents), rng))
    # The final 20% is untouched until after candidate generation and is report-only.
    champion = scored[0]
    test = _evaluate(candles, champion["genome"], validation_end, n)
    val = champion["validation"]
    passed = (val["samples"] >= 20 and test["samples"] >= 20
              and val["directional_accuracy"] > 0.5
              and test["directional_accuracy"] > 0.5
              and val["mean_abs_correlation"] > 0
              and test["mean_abs_correlation"] > 0)
    next_memory = {
        "generation": int(memory.get("generation", 0))+1,
        "archive": next_population[:12],
        "last_test": test,
        "last_validation": val,
        "last_validation_passed": bool(passed),
        "updated_at": int(candles[-1].get("ts", 0)),
    }
    return {
        "memory": next_memory,
        "report": {
            "status": "research_candidate_passed" if passed else "research_only",
            "generation": next_memory["generation"],
            "population": len(population),
            "best_input_recipe": champion["genome"],
            "development": champion["train"],
            "validation": val,
            "untouched_test": test,
            "validation_passed": bool(passed),
            "live_schema_changed": False,
            "production_model_changed": False,
        },
    }
