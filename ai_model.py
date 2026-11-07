"""Thirty-layer signal model plus a learned TP/SL exit policy."""
import math
import os

from feature_engine import FEATURE_VERSION

MODEL_PATH = "xrp_model.pt"
EXIT_MODEL_PATH = "xrp_exit_policy.pt"
GATE_MIN_WINS = 30
GATE_MIN_LOSSES = 30
GATE_CLOSED = GATE_MIN_WINS + GATE_MIN_LOSSES
MODEL_ARCH_VERSION = "deep30x128-v1"
EXIT_ARCH_VERSION = "exit30x128-v1"
FEATURE_DIM = 1205

try:
    import torch
    from torch import nn

    class DeepSignalModel(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM):
            super().__init__()
            layers = []
            dim = input_dim
            for _ in range(30):
                layers.extend([nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.08)])
                dim = 128
            layers.append(nn.Linear(dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class ExitPolicyModel(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM):
            super().__init__()
            layers = []
            dim = input_dim
            for _ in range(30):
                layers.extend([nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.08)])
                dim = 128
            layers.append(nn.Linear(dim, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return torch.nn.functional.softplus(self.net(x))
except ImportError:
    torch = None
    DeepSignalModel = None
    ExitPolicyModel = None
    nn = None


def _outcome(item):
    try:
        net = float(item.get("net_return_pct"))
        if math.isfinite(net):
            return 1 if net > 0 else 0
    except (TypeError, ValueError):
        pass
    return 1 if item.get("status") == "TP" else 0


def _valid_features(item):
    features = item.get("features")
    if not isinstance(features, dict) or len(features) != FEATURE_DIM:
        return False
    if item.get("feature_version") not in (None, FEATURE_VERSION):
        return False
    try:
        return all(math.isfinite(float(value)) for value in features.values())
    except (TypeError, ValueError):
        return False


def _excursion_targets(item):
    """Return observed maximum favorable/adverse excursion in percent."""
    try:
        entry = float(item.get("entry", 0))
        high = float(item.get("high_water", item.get("mfe_high", 0)))
        low = float(item.get("low_water", item.get("mae_low", 0)))
        if not all(math.isfinite(x) and x > 0 for x in (entry, high, low)):
            return None
        mfe = max(0.0, (high / entry - 1.0) * 100.0)
        mae = max(0.0, (1.0 - low / entry) * 100.0)
        if not (math.isfinite(mfe) and math.isfinite(mae)):
            return None
        return min(10.0, max(0.02, mfe)), min(10.0, max(0.02, mae))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def gate_status(history):
    closed = [x for x in history if x.get("status") in ("TP", "SL")]
    compatible = [x for x in closed if _valid_features(x)]
    wins = sum(1 for item in compatible if _outcome(item) == 1)
    losses = len(compatible) - wins
    exit_samples = sum(1 for item in compatible if _excursion_targets(item) is not None)
    # A balanced gate prevents a model trained almost entirely on one outcome
    # from being treated as decision-ready.
    enabled = wins >= GATE_MIN_WINS and losses >= GATE_MIN_LOSSES and exit_samples >= GATE_CLOSED
    return {
        "closed_signals": len(closed),
        "compatible_signals": len(compatible),
        "required_closed": GATE_CLOSED,
        "required_wins": GATE_MIN_WINS,
        "required_losses": GATE_MIN_LOSSES,
        "wins": wins,
        "losses": losses,
        "enabled": enabled,
        "exit_policy_samples": exit_samples,
        "exit_policy_required": GATE_CLOSED,
        "gate_reason": "balanced_sample_ready" if enabled else "requires_30_compatible_wins_and_30_losses",
    }


def _load_checkpoint(path, architecture):
    if torch is None or not os.path.isfile(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if (data.get("feature_version") != FEATURE_VERSION
                or data.get("architecture") != architecture
                or int(data.get("input_dim", 0)) != FEATURE_DIM
                or not isinstance(data.get("feature_mean"), list)
                or not isinstance(data.get("feature_std"), list)
                or len(data["feature_mean"]) != FEATURE_DIM
                or len(data["feature_std"]) != FEATURE_DIM
                or int(data.get("trained_wins", 0)) < GATE_MIN_WINS
                or int(data.get("trained_losses", 0)) < GATE_MIN_LOSSES):
            return None
        return data
    except Exception:
        return None


def _normalizer(raw_train, previous):
    if previous:
        mean = torch.tensor(previous["feature_mean"], dtype=torch.float32)
        std = torch.tensor(previous["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    else:
        mean = raw_train.mean(dim=0)
        std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _train_classifier(eligible, previous):
    keys = sorted(eligible[0]["features"])
    if any(sorted(item["features"]) != keys for item in eligible):
        return None, "feature schema mismatch"
    rows = [[float(item["features"][key]) for key in keys] for item in eligible]
    labels = [_outcome(item) for item in eligible]
    split = max(1, int(len(rows) * 0.8))
    if split >= len(rows):
        split = len(rows) - 1
    raw_train = torch.tensor(rows[:split], dtype=torch.float32)
    raw_val = torch.tensor(rows[split:], dtype=torch.float32)
    y = torch.tensor(labels[:split], dtype=torch.float32).view(-1, 1)
    yv = torch.tensor(labels[split:], dtype=torch.float32).view(-1, 1)
    mean, std = _normalizer(raw_train, previous)
    x = ((raw_train - mean) / std).clamp(-10.0, 10.0)
    xv = ((raw_val - mean) / std).clamp(-10.0, 10.0)
    model = DeepSignalModel(len(keys))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if previous:
        try:
            model.load_state_dict(previous["state_dict"])
            if previous.get("optimizer_state_dict"):
                optimizer.load_state_dict(previous["optimizer_state_dict"])
        except Exception:
            model = DeepSignalModel(len(keys))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_logits = model(xv) if len(xv) else None
        val_loss = float(loss_fn(val_logits, yv)) if val_logits is not None else None
        val_probs = torch.sigmoid(val_logits) if val_logits is not None else None
        val_accuracy = float(((val_probs >= 0.5) == (yv >= 0.5)).float().mean()) if val_probs is not None else None
        val_brier = float(((val_probs - yv) ** 2).mean()) if val_probs is not None else None
    torch.save({
        "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": len(keys), "feature_version": FEATURE_VERSION,
        "architecture": MODEL_ARCH_VERSION, "feature_mean": mean.tolist(),
        "feature_std": std.tolist(), "trained_samples": len(rows),
        "trained_wins": sum(labels), "trained_losses": len(labels) - sum(labels),
    }, MODEL_PATH)
    return {
        "trained": True, "samples": len(rows), "train_samples": len(x),
        "validation_samples": len(xv), "train_loss": float(loss.detach()),
        "validation_loss": val_loss, "validation_accuracy": val_accuracy,
        "validation_brier": val_brier, "resumed_previous_model": bool(previous),
    }, None


def _train_exit_policy(eligible, previous):
    samples = [(item, _excursion_targets(item)) for item in eligible]
    samples = [(item, target) for item, target in samples if target is not None]
    if len(samples) < GATE_CLOSED:
        return {"trained": False, "reason": "60 balanced compatible closed signals with observed price excursions required",
                "samples": len(samples)}
    keys = sorted(samples[0][0]["features"])
    if any(sorted(item["features"]) != keys for item, _ in samples):
        return {"trained": False, "reason": "feature schema mismatch", "samples": len(samples)}
    rows = [[float(item["features"][key]) for key in keys] for item, _ in samples]
    targets = [target for _, target in samples]
    split = max(1, int(len(rows) * 0.8))
    if split >= len(rows):
        split = len(rows) - 1
    raw_train = torch.tensor(rows[:split], dtype=torch.float32)
    raw_val = torch.tensor(rows[split:], dtype=torch.float32)
    y = torch.tensor(targets[:split], dtype=torch.float32)
    yv = torch.tensor(targets[split:], dtype=torch.float32)
    mean, std = _normalizer(raw_train, previous)
    x = ((raw_train - mean) / std).clamp(-10.0, 10.0)
    xv = ((raw_val - mean) / std).clamp(-10.0, 10.0)
    model = ExitPolicyModel(len(keys))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if previous:
        try:
            model.load_state_dict(previous["state_dict"])
            if previous.get("optimizer_state_dict"):
                optimizer.load_state_dict(previous["optimizer_state_dict"])
        except Exception:
            model = ExitPolicyModel(len(keys))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = float(loss_fn(model(xv), yv)) if len(xv) else None
    torch.save({
        "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": len(keys), "feature_version": FEATURE_VERSION,
        "architecture": EXIT_ARCH_VERSION, "feature_mean": mean.tolist(),
        "feature_std": std.tolist(), "trained_samples": len(rows),
        "trained_wins": sum(_outcome(item) for item, _ in samples),
        "trained_losses": len(samples) - sum(_outcome(item) for item, _ in samples),
    }, EXIT_MODEL_PATH)
    return {"trained": True, "samples": len(rows), "train_samples": len(x),
            "validation_samples": len(xv), "train_loss": float(loss.detach()),
            "validation_loss": val_loss, "resumed_previous_model": bool(previous)}


def train(history, epochs=2):
    gate = gate_status(history)
    if torch is None:
        return {"trained": False, "reason": "torch unavailable", "gate": gate}
    if not gate["enabled"]:
        return {"trained": False, "reason": "at least 30 compatible wins and 30 compatible losses required", "gate": gate}
    eligible = [item for item in history
                if item.get("status") in ("TP", "SL") and _valid_features(item)]
    previous = _load_checkpoint(MODEL_PATH, MODEL_ARCH_VERSION)
    classifier, error = _train_classifier(eligible, previous)
    if error:
        return {"trained": False, "reason": error, "gate": gate}
    exit_previous = _load_checkpoint(EXIT_MODEL_PATH, EXIT_ARCH_VERSION)
    exit_policy = _train_exit_policy(eligible, exit_previous)
    classifier["exit_policy"] = exit_policy
    classifier["trained"] = True
    classifier["gate"] = gate
    classifier["warning"] = "Early-stage model: small sample size; probability and exit levels are experimental."
    return classifier


def _predict(features, path, architecture, model_class):
    if torch is None or not isinstance(features, dict) or len(features) != FEATURE_DIM:
        return None
    data = _load_checkpoint(path, architecture)
    if data is None:
        return None
    try:
        keys = sorted(features)
        values = [float(features[key]) for key in keys]
        if not all(math.isfinite(value) for value in values):
            return None
        model = model_class(len(keys))
        model.load_state_dict(data["state_dict"])
        model.eval()
        mean = torch.tensor(data["feature_mean"], dtype=torch.float32)
        std = torch.tensor(data["feature_std"], dtype=torch.float32).clamp_min(1e-6)
        sample = torch.tensor([values], dtype=torch.float32)
        sample = ((sample - mean) / std).clamp(-10.0, 10.0)
        with torch.no_grad():
            output = model(sample)
        return output[0].tolist() if model_class is ExitPolicyModel else float(torch.sigmoid(output).item())
    except Exception:
        return None


def predict_probability(features):
    value = _predict(features, MODEL_PATH, MODEL_ARCH_VERSION, DeepSignalModel)
    return value if isinstance(value, float) and math.isfinite(value) else None


def predict_exit_policy(features, entry_price, atr_pct=0.0):
    """Predict bounded TP/SL levels from learned MFE/MAE behavior."""
    raw = _predict(features, EXIT_MODEL_PATH, EXIT_ARCH_VERSION, ExitPolicyModel)
    if raw is None:
        return None
    try:
        mfe, mae = (float(raw[0]), float(raw[1]))
        if not all(math.isfinite(x) for x in (mfe, mae)) or entry_price <= 0:
            return None
        # Blend learned excursion estimates with an ATR floor so the model cannot
        # place exits inside ordinary market noise. Keep strict risk bounds.
        atr_pct = max(0.0, float(atr_pct or 0.0))
        tp_floor = min(6.0, max(2.0, atr_pct * 2.0))
        sl_floor = min(3.0, max(1.0, atr_pct * 1.20))
        tp_pct = min(6.0, max(tp_floor, mfe * 0.75))
        sl_pct = min(3.0, max(sl_floor, mae * 0.85))
        tp_pct = min(6.0, max(tp_pct, sl_pct * 1.50))
        return {
            "tp": round(float(entry_price) * (1.0 + tp_pct / 100.0), 10),
            "sl": round(float(entry_price) * (1.0 - sl_pct / 100.0), 10),
            "tp_distance_pct": round(tp_pct, 5),
            "sl_distance_pct": round(sl_pct, 5),
            "predicted_mfe_pct": round(mfe, 5),
            "predicted_mae_pct": round(mae, 5),
            "source": "AI_LEARNED_EXIT_POLICY",
            "atr_pct": round(atr_pct, 5),
            "exit_distance_source": "AI_LEARNED_PLUS_ATR_GUARD",
            "model_samples": int(_load_checkpoint(EXIT_MODEL_PATH, EXIT_ARCH_VERSION).get("trained_samples", 0)),
        }
    except (TypeError, ValueError, ZeroDivisionError):
        return None


ENTRY_METHODS = ("MARKET", "BREAKOUT", "RETEST", "SUPPORT_BOUNCE")

def select_entry_method(market_context, method_stats=None):
    """Online, outcome-based entry policy with regime-aware priors and exploration."""
    ctx = market_context or {}
    stats = method_stats or {}
    regime = ctx.get("regime", "range")
    trend = float(ctx.get("trend", 0.0) or 0.0)
    volatility = float(ctx.get("volatility", 0.0) or 0.0)
    imbalance = float(ctx.get("book_imbalance", 0.0) or 0.0)
    priors = {
        "MARKET": 0.48 + (0.12 if trend > 0.003 else 0.0) - (0.08 if volatility > 0.008 else 0.0),
        "BREAKOUT": 0.55 if regime == "bullish" else 0.30,
        "RETEST": 0.58 if regime == "bullish" else (0.42 if regime == "range" else 0.25),
        "SUPPORT_BOUNCE": 0.55 if regime == "range" else (0.35 if regime == "bearish" else 0.43),
    }
    if imbalance > 0.15:
        priors["MARKET"] += 0.04
        priors["BREAKOUT"] += 0.03
    if volatility > 0.008:
        priors["RETEST"] += 0.08
        priors["SUPPORT_BOUNCE"] += 0.04
    scored = {}
    total_entry_samples = sum(int((stats.get(name) or {}).get("trades", 0) or 0) for name in ENTRY_METHODS)
    learning_ready = total_entry_samples >= GATE_CLOSED
    for name in ENTRY_METHODS:
        row = stats.get(name, {})
        n = int(row.get("trades", 0) or 0)
        avg = float(row.get("avg_net_return_pct", 0.0) or 0.0)
        win_rate = float(row.get("win_rate_pct", 50.0) or 50.0)
        # Shrink small samples toward the regime prior; historical performance
        # becomes influential only as real closed trades accumulate.
        weight = min(0.70, n / 30.0) if learning_ready else 0.0
        learned = 0.5 + max(-0.5, min(0.5, avg / 2.0)) * 0.55 + (win_rate - 50.0) / 100.0 * 0.45
        scored[name] = (1.0 - weight) * priors[name] + weight * learned
    chosen = max(ENTRY_METHODS, key=lambda name: (scored[name], name))
    return {
        "method": chosen,
        "scores": {name: round(value, 5) for name, value in scored.items()},
        "samples": {name: int((stats.get(name) or {}).get("trades", 0) or 0) for name in ENTRY_METHODS},
        "total_samples": total_entry_samples, "learning_ready": learning_ready,
        "source": "ONLINE_LEARNED_ENTRY_POLICY" if learning_ready else "REGIME_PRIOR_UNTIL_10_CLOSED",
    }
