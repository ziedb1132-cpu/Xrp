"""Bounded genetic mutation lab for XRP strategy research (never executes orders)."""
import copy
import math
import random

from strategy_engine import (
    STRATEGIES, _ema, _rsi, ROUND_TRIP_COST_PCT,
)

FAMILIES = tuple(name for name, _ in STRATEGIES)
POPULATION_SIZE = 16
GENERATIONS = 3
MAX_ARCHIVE = 24
MIN_VALIDATION_TRADES = 5


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def _normalize(genome):
    g = dict(genome or {})
    family = g.get("family", FAMILIES[0])
    if family not in FAMILIES:
        family = FAMILIES[0]
    sl = _clamp(g.get("sl_pct", 1.5), 1.0, 3.0)
    tp = _clamp(g.get("tp_pct", 3.0), max(2.0, sl * 1.5), 6.0)
    return {
        "family": family,
        "momentum_min": _clamp(g.get("momentum_min", 0.0015), 0.0005, 0.006),
        "volume_min": _clamp(g.get("volume_min", 1.25), 1.0, 2.5),
        "rsi_low": _clamp(g.get("rsi_low", 38), 25, 50),
        "rsi_high": _clamp(g.get("rsi_high", 65), 55, 78),
        "tp_pct": tp,
        "sl_pct": sl,
        "trail_activation_pct": _clamp(g.get("trail_activation_pct", 35), 20, 70),
        "trail_gap_pct": _clamp(g.get("trail_gap_pct", 0.7), 0.25, 1.5),
    }


def _entry(candles, i, g):
    window = candles[:i + 1]
    closes = [float(x["c"]) for x in window]
    volumes = [float(x.get("vol", 0)) for x in window]
    if len(closes) < 40:
        return False
    last, previous = closes[-1], closes[-2]
    momentum = last / closes[-7] - 1 if closes[-7] else 0.0
    avg_volume = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else 0.0
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0.0
    rsi = _rsi(closes)
    family = g["family"]
    if family == "EMA_8_21_TREND":
        base = _ema(closes[-40:], 8) > _ema(closes[-40:], 21) and momentum > 0
    elif family == "EMA_13_34_TREND":
        base = _ema(closes[-40:], 13) > _ema(closes[-40:], 34) and momentum > 0
    elif family == "MOMENTUM_6":
        base = momentum >= g["momentum_min"]
    elif family == "BREAKOUT_20":
        base = last > max(float(x["h"]) for x in window[-21:-1])
    elif family == "RSI_RECOVERY":
        base = g["rsi_low"] < rsi < g["rsi_high"] and last > previous
    elif family == "VOLUME_BREAKOUT":
        base = last > previous and volume_ratio >= g["volume_min"]
    elif family == "VWAP_RECLAIM":
        typical = [(float(x["h"]) + float(x["l"]) + float(x["c"])) / 3 for x in window[-20:]]
        vols = volumes[-20:]
        vwap = sum(p * v for p, v in zip(typical, vols)) / max(1e-12, sum(vols))
        prev_typical = (float(window[-2]["h"]) + float(window[-2]["l"]) + float(window[-2]["c"])) / 3
        base = prev_typical <= vwap and last > vwap
    else:  # RANGE_EXPANSION
        prev_range = float(window[-2]["h"]) - float(window[-2]["l"])
        last_range = float(window[-1]["h"]) - float(window[-1]["l"])
        base = last > previous and last_range > 1.25 * max(prev_range, 1e-12)
    return bool(base and momentum >= -0.001 and volume_ratio >= 0.5)


def _simulate(candles, genome):
    g = _normalize(genome)
    trades = []
    i = 40
    while i < len(candles) - 1:
        if not _entry(candles, i, g):
            i += 1
            continue
        entry = float(candles[i]["c"])
        tp = entry * (1 + g["tp_pct"] / 100)
        sl = entry * (1 - g["sl_pct"] / 100)
        high_water = entry
        exit_price = None
        reason = None
        exit_index = None
        for j in range(i + 1, len(candles)):
            bar = candles[j]
            low, high = float(bar["l"]), float(bar["h"])
            # Conservative ordering when OHLC cannot identify intrabar path.
            if low <= sl:
                exit_price, reason, exit_index = sl, "SL", j
                break
            if high >= tp:
                exit_price, reason, exit_index = tp, "TP", j
                break
            high_water = max(high_water, float(bar["c"]))
            activation = entry * (1 + g["tp_pct"] * g["trail_activation_pct"] / 10000)
            if high_water >= activation:
                proposed = max(entry * (1 + ROUND_TRIP_COST_PCT / 100),
                               high_water * (1 - g["trail_gap_pct"] / 100))
                sl = max(sl, proposed)
        if exit_price is None:
            break
        gross = (exit_price / entry - 1) * 100
        trades.append({"net_pct": gross - ROUND_TRIP_COST_PCT, "reason": reason})
        i = exit_index + 1
    values = [x["net_pct"] for x in trades]
    wins = [x for x in values if x > 0]
    losses = [x for x in values if x < 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss else (99.0 if gross_win else 0.0)
    return {
        "trades": len(values), "wins": len(wins), "losses": len(losses),
        "net_profit_pct": round(sum(values), 5),
        "expectancy_pct": round(sum(values) / len(values), 5) if values else None,
        "profit_factor": round(pf, 4),
        "win_rate_pct": round(100 * len(wins) / len(values), 2) if values else None,
    }


def _fitness(metrics):
    expectancy = metrics.get("expectancy_pct")
    if expectancy is None or metrics["trades"] < 2:
        return -999.0
    # Trade count contributes modestly; profit quality dominates.
    return round(expectancy * min(metrics["trades"], 20) ** 0.5
                 + min(metrics["profit_factor"], 3.0) * 0.05, 6)


def _random_genome(rng):
    return _normalize({
        "family": rng.choice(FAMILIES),
        "momentum_min": rng.uniform(0.0005, 0.006),
        "volume_min": rng.uniform(1.0, 2.5),
        "rsi_low": rng.uniform(25, 50),
        "rsi_high": rng.uniform(55, 78),
        "tp_pct": rng.uniform(2.0, 6.0),
        "sl_pct": rng.uniform(1.0, 3.0),
        "trail_activation_pct": rng.uniform(20, 70),
        "trail_gap_pct": rng.uniform(0.25, 1.5),
    })


def _mutate(parent, rng):
    child = copy.deepcopy(parent)
    if rng.random() < 0.35:
        child["family"] = rng.choice(FAMILIES)
    for key, scale in (("momentum_min", 0.001), ("volume_min", 0.25),
                       ("rsi_low", 5), ("rsi_high", 5), ("tp_pct", 0.8),
                       ("sl_pct", 0.4), ("trail_activation_pct", 8),
                       ("trail_gap_pct", 0.2)):
        if rng.random() < 0.45:
            child[key] = float(child[key]) + rng.gauss(0, scale)
    return _normalize(child)


def _crossover(a, b, rng):
    child = {key: (a[key] if rng.random() < 0.5 else b[key]) for key in a}
    return _normalize(child)


def evolve_population(candles, memory=None):
    """Evolve candidate strategy/exit combinations; promotion is research-only."""
    memory = dict(memory or {})
    if len(candles) < 100:
        return {"memory": memory, "report": {"status": "insufficient_candles",
                "candles": len(candles), "population": 0, "promoted": False}}
    seed = int(candles[-1].get("ts", len(candles))) ^ int(memory.get("generation", 0))
    rng = random.Random(seed)
    archive = [_normalize(x) for x in memory.get("archive", []) if isinstance(x, dict)]
    population = archive[:8]
    while len(population) < POPULATION_SIZE:
        population.append(_random_genome(rng))
    split = max(60, min(len(candles) - 30, int(len(candles) * 0.65)))
    train_data, validation_data = candles[:split], candles[split:]
    best = None
    evaluated = []
    for generation in range(GENERATIONS):
        scored = []
        for genome in population:
            train = _simulate(train_data, genome)
            validation = _simulate(validation_data, genome)
            score = _fitness(validation)
            candidate = {"genome": genome, "train": train, "validation": validation,
                         "fitness": score}
            scored.append(candidate)
            evaluated.append(candidate)
        scored.sort(key=lambda x: x["fitness"], reverse=True)
        if best is None or scored[0]["fitness"] > best["fitness"]:
            best = scored[0]
        parents = [x["genome"] for x in scored[:max(2, POPULATION_SIZE // 4)]]
        next_population = parents[:]
        while len(next_population) < POPULATION_SIZE:
            p1, p2 = rng.choice(parents), rng.choice(parents)
            next_population.append(_mutate(_crossover(p1, p2, rng), rng))
        population = next_population
    v = best["validation"]
    promoted = (v["trades"] >= MIN_VALIDATION_TRADES
                and v["net_profit_pct"] > 0
                and v["expectancy_pct"] is not None and v["expectancy_pct"] > 0
                and v["profit_factor"] > 1.05
                and best["train"]["net_profit_pct"] > 0
                and best["train"]["profit_factor"] > 1.0)
    candidates = sorted(evaluated, key=lambda x: x["fitness"], reverse=True)
    unique = []
    seen = set()
    for item in candidates:
        key = tuple(sorted(item["genome"].items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item["genome"])
        if len(unique) >= MAX_ARCHIVE:
            break
    next_memory = {
        "generation": int(memory.get("generation", 0)) + GENERATIONS,
        "archive": unique,
        "last_validation": v,
        "last_train": best["train"],
        "last_validation_passed": bool(promoted),
        "updated_at": int(candles[-1].get("ts", 0)),
    }
    report = {
        "status": "validation_passed" if promoted else "research_only",
        "generation": next_memory["generation"],
        "population": POPULATION_SIZE,
        "generations_this_cycle": GENERATIONS,
        "unique_candidates": len(unique),
        "best_family": best["genome"]["family"],
        "best_genome": best["genome"],
        "train": best["train"],
        "validation": v,
        "validation_passed": bool(promoted),
        "promotion_rule": "validation>=5 trades, positive net expectancy and profit, PF>1.05; train also positive",
        "live_execution_enabled": False,
    }
    return {"memory": next_memory, "report": report}
