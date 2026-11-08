"""
stylemodel.py - learned opponent-style advisor for the EV (Bayesian range) strategy.

Every REFRESH hands it turns the opponent's observed action frequencies (the counts the
Bayesian range model already keeps) into a feature vector and classifies the opponent's style
(aggressive / normal / nit) by distance to style centroids learned from 32 real public bots
(k-means on 905 logged halves; leave-one-bot-out accuracy 91% at 60 hands, 94% at 240). The
style probabilities blend per-style exploit parameters, so they move with the evidence.

Data: data/style_centroids.json (centroids, scaler, temperature), data/style_profiles.json
(per-style overrides).
Numpy only. Any problem -> the caller's default parameters.
"""
from __future__ import annotations

import json
import os

import numpy as np

import range_bayes as RB

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_HANDS = 60        # first classification after this many hands (91% accuracy on unseen real bots)
REFRESH = 30          # then every REFRESH hands
PN = 10.0             # smoothing pseudo-observations per feature
FEATURE_NAMES = ["bet_free", "raise_face", "fold_face", "reraise_after_bet", "fold_after_bet",
                 "pf_raise", "pf_fold", "river_bet_free", "sd_post_strength", "log_hands"]
_PRIOR_CNT = RB.new_state()["cnt"]


def _rate(k: float, n: float, prior: float, pn: float = PN) -> float:
    return float((k + prior * pn) / (n + pn))


def features(m: dict, hands: int) -> np.ndarray:
    """Style features in [0, 1] from the strategy model dict (observed counts only)."""
    raw = np.maximum(0.0, m["rb"]["cnt"] - _PRIOR_CNT)      # (street, ctx, action)
    post = raw[1:].sum(axis=0)
    x = [
        _rate(post[0, 2], post[0].sum(), 0.35),             # bets when checked to
        _rate(post[1, 2], post[1].sum(), 0.15),             # raises facing our bet
        _rate(post[1, 0], post[1].sum(), 0.35),             # folds facing our bet
        _rate(post[2, 2], post[2].sum(), 0.20),             # re-raises our raise after betting
        _rate(post[2, 0], post[2].sum(), 0.25),             # folds to our raise after betting
        _rate(raw[0, :, 2].sum(), raw[0].sum(), 0.30),      # preflop raises
        _rate(raw[0, 1, 0], raw[0, 1].sum(), 0.20),         # preflop folds facing a bet
        _rate(raw[3, 0, 2], raw[3, 0].sum(), 0.35),         # river bets when checked to
        _rate(m["sd_raise_pct"][1], m["sd_raise_n"][1], 0.5, 3.0),   # strength of shown raises
        float(np.log1p(hands) / np.log(1001.0)),
    ]
    return np.array(x, dtype=np.float64)


def blend_params(default: dict, overrides: list, probs) -> dict:
    """Probability-weighted average of {default + override_k} over the overridden numeric keys."""
    out = dict(default)
    keys = {k for o in overrides for k in o}
    for key in keys:
        base = default.get(key, 0.0)
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            continue
        vals = [float(o.get(key, base)) for o in overrides]
        out[key] = float(sum(p * v for p, v in zip(probs, vals)))
    return out


def _load():
    """Style centroids learned from 32 real public bots (k-means on full-half behaviour) and the
    per-style parameter overrides. Returns None on any problem (caller keeps its defaults)."""
    try:
        with open(os.path.join(HERE, "data", "style_centroids.json"), encoding="utf-8") as f:
            c = json.load(f)
        with open(os.path.join(HERE, "data", "style_profiles.json"), encoding="utf-8") as f:
            prof = json.load(f)["styles"]
        dims = np.array(c["features"], dtype=int)
        mu, sd, C = (np.array(c[k], dtype=np.float64) for k in ("mu", "sd", "centroids"))
        names = list(c["names"])
        if C.shape != (len(names), len(dims)) or not (sd > 0).all() or not np.isfinite(C).all():
            return None
        overrides = [dict(prof.get(n, {})) for n in names]
        return dims, mu, sd, C, float(c.get("temperature", 2.0)), names, overrides
    except Exception:
        return None


_STATE = _load()


def classify(m: dict, hands: int, state=None):
    """Style probabilities (dict name -> p) from the observed features, or None."""
    st = state if state is not None else _STATE
    if st is None:
        return None
    dims, mu, sd, C, T, names, _ = st
    z = (features(m, hands)[dims] - mu) / sd
    d2 = ((z[None, :] - C) ** 2).sum(axis=1)
    p = np.exp(-(d2 - d2.min()) / T)
    p /= p.sum()
    return dict(zip(names, p.tolist()))


class Advisor:
    """Per-half advisor; keep one in the strategy model dict. Re-classifies every REFRESH hands
    from MIN_HANDS on and blends the per-style parameter overrides by the style probabilities."""

    def __init__(self, default: dict, state=None):
        self.default = default
        self.cache = default
        self.last = -10 ** 9
        self.probs = None
        self.state = state if state is not None else _STATE

    def params(self, m: dict, hands: int) -> dict:
        try:
            if self.state is None or hands < MIN_HANDS:
                return self.default
            if 0 <= hands - self.last < REFRESH:
                return self.cache
            self.last = hands
            probs = classify(m, hands, self.state)
            if probs is None or not all(np.isfinite(v) for v in probs.values()):
                self.cache = self.default
                return self.cache
            self.probs = probs
            names, overrides = self.state[5], self.state[6]
            self.cache = blend_params(self.default, overrides, [probs[n] for n in names])
            return self.cache
        except Exception:
            self.cache = self.default
            self.last = hands
            return self.default
