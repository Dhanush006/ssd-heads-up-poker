"""
oppmodel.py - runtime opponent-style advisor for the Short Deck PLO5 bot.

Reads the counters strategy.py already keeps in memory["model"], turns them into
8 features, runs a tiny softmax classifier over K strategy profiles, and returns a
probability-weighted blend of parameter dicts for strategy.decide().

Time budget: one feature+predict+blend pass is about 55 us; it runs only every
REFRESH hands after MIN_HANDS. Numpy only (no torch/sklearn import).
Any problem -> the caller gets the default parameters back.

Blending rules: keys in NO_BLEND (n_samples, prior_n, prior_raise, prior_fold, eps)
feed the opponent model's own smoothing and always keep the default value; int-typed
defaults are never blended either. Returned dicts are shared (cached): treat as read-only.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_HANDS = 40        # no adaptation before this many finished hands
REFRESH = 20          # recompute at most every N hands
MIN_CONF = 0.45       # top profile probability needed to leave the defaults
PRIOR_N = 10.0

NO_BLEND = frozenset({"n_samples", "prior_n", "prior_raise", "prior_fold", "eps"})

FEATURE_NAMES = ["pf_raise", "flop_raise", "turn_raise", "river_raise",
                 "fold_to_raise", "sd_pre", "sd_post", "log_hands"]


def _smooth(k: float, n: float, prior: float, pn: float = PRIOR_N) -> float:
    return (k + prior * pn) / (n + pn)


def features(m: dict, hands: int) -> np.ndarray:
    """8 features in [0, 1] from the strategy model dict."""
    act, rai = m["opp_act"], m["opp_raise"]
    x = [_smooth(rai[s], act[s], 0.30) for s in range(4)]
    x.append(_smooth(m["fold_to_raise"], m["faced"], 0.35))
    for k in (0, 1):
        x.append(_smooth(m["sd_raise_pct"][k], m["sd_raise_n"][k], 0.5, 3.0))
    x.append(float(np.log1p(hands) / np.log(1001.0)))
    return np.array(x, dtype=np.float64)


def predict(x: np.ndarray, W: np.ndarray, b: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Softmax probabilities over profiles."""
    z = W @ ((x - mu) / sd) + b
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def blend_params(default: dict, overrides: list, probs: np.ndarray) -> dict:
    """Probability-weighted average of {default + override_k} for numeric params."""
    profiles = [{**default, **o} for o in overrides]
    out = dict(default)
    for key, base in default.items():
        if key in NO_BLEND or (isinstance(base, int) and not isinstance(base, bool)):
            continue
        if isinstance(base, list):
            out[key] = [float(sum(p * prof[key][i] for p, prof in zip(probs, profiles)))
                        for i in range(len(base))]
        elif isinstance(base, (int, float)) and not isinstance(base, bool):
            out[key] = float(sum(p * prof[key] for p, prof in zip(probs, profiles)))
    return out


def _finite_nums(v) -> bool:
    if isinstance(v, (list, tuple)):
        return all(_finite_nums(e) for e in v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(np.isfinite(v))
    return True


def _valid(W, b, mu, sd, overrides) -> bool:
    n = len(FEATURE_NAMES)
    K = len(overrides)
    if W.shape != (K, n) or b.shape != (K,) or mu.shape != (n,) or sd.shape != (n,):
        return False
    if not all(np.isfinite(a).all() for a in (W, b, mu, sd)) or not (sd > 0).all():
        return False
    return all(_finite_nums(v) for o in overrides for v in o.values())


def _load():
    try:
        w = np.load(os.path.join(HERE, "data", "oppmodel.npz"))
        W, b, mu, sd = (np.array(w[k], dtype=np.float64) for k in ("W", "b", "mu", "sd"))
        with open(os.path.join(HERE, "data", "profiles.json"), encoding="utf-8") as f:
            overrides = json.load(f)["overrides"]
        if not _valid(W, b, mu, sd, overrides):
            return None
        return W, b, mu, sd, overrides
    except Exception:
        return None


_STATE = _load()


class Advisor:
    """Per-half advisor. Keep one in memory["adv"]; it holds the cached blend."""

    def __init__(self, default: dict):
        self.default = default
        self.cache = default
        self.last = -10**9
        self.probs = None

    def params(self, m: dict, hands: int) -> dict:
        try:
            if _STATE is None or hands < MIN_HANDS:
                return self.default
            if 0 <= hands - self.last < REFRESH:
                return self.cache
            W, b, mu, sd, overrides = _STATE
            probs = predict(features(m, hands), W, b, mu, sd)
            self.probs, self.last = probs, hands
            if not np.isfinite(probs).all() or float(probs.max()) < MIN_CONF:
                self.cache = self.default
            else:
                self.cache = blend_params(self.default, overrides, probs)
            return self.cache
        except Exception:
            self.cache = self.default
            self.last = hands
            return self.default
