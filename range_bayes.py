"""range_bayes.py - Bayesian opponent range from every action this hand.

Ideas taken from the public sparring bots that beat v22:

* Each opponent action is evidence about its hand. P(action | strength x) is an ordered
  logistic in x (x = strength percentile on the street of the action, 0 weak .. 1 strong):
      P(raise | x) = sig(a_r + beta (x - 1/2)),  P(fold | x) = sig(a_f - beta (x - 1/2)) (facing a bet)
      P(check/call | x) = the rest.
  Intercepts a_* are set in closed form so the average over x matches the opponent's observed
  frequency of that action in this (street, context) - context 0 = nothing to call (check or
  bet), 1 = facing our bet/raise. Checks are evidence too (they cap the range).
* beta = how much the opponent's actions track its cards. It is unknown and differs by
  opponent and street, so each street keeps a posterior over BETAS, updated at showdown from
  the actions of the hand we then see (likelihood of their real strength under each beta,
  relative to a random hand).
* Each sampled opponent hand gets log w = sum over streets of log( sum_beta pi[s, beta] *
  prod_{actions on s} P(action | x_s, beta) ). Equity is the w-weighted win share.

All functions are pure numpy and work on the samples strategy.equity draws.
"""
from __future__ import annotations

import math

import numpy as np

# Model grid: informativeness beta x bluff share b of the raising range. The bluff component
# (polarised ranges) is off by default: it helped against one sparring bot but cost -0.24 bb/deal overall.
_BETA_GRID = np.array([0.0, 2.0, 4.0, 8.0, 16.0])
_BETA_PRIOR = np.array([0.15, 0.20, 0.25, 0.25, 0.15])


def configure(bluff: bool = False) -> None:
    """Choose the model grid (call before new_state)."""
    global BETAS, BLUFFS, PRIOR_PI
    grid, prior = (np.array([0.0, 0.25, 0.5]), np.array([0.5, 0.3, 0.2])) if bluff else (np.array([0.0]), np.array([1.0]))
    BETAS = np.repeat(_BETA_GRID, len(grid))
    BLUFFS = np.tile(grid, len(_BETA_GRID))
    PRIOR_PI = np.outer(_BETA_PRIOR, prior).ravel()


configure(False)
FLOOR = 0.02          # no action ever has probability below this (misclicks, mixed strategies)
FOLD, CALL, RAISE = 0, 1, 2


from functools import lru_cache


@lru_cache(maxsize=8192)
def _intercept_cached(p: float, beta: float) -> float:
    return intercept(p, beta)


def _icpt(p: float, beta: float) -> float:
    return _intercept_cached(round(p, 4), round(beta, 3))


def intercept(p: float, beta: float) -> float:
    """Logit intercept a with mean over uniform x of sig(a + beta (x - 1/2)) equal to p."""
    p = min(1.0 - 1e-9, max(1e-9, p))
    if beta < 1e-7:
        return math.log(p / (1.0 - p))
    return math.log(math.expm1(beta * p)) - math.log(-math.expm1(beta * (p - 1.0))) - beta / 2.0


def new_state() -> dict:
    # Dirichlet pseudo-counts [fold, call/check, raise] per street and context.
    # ctx 0 (nothing to call): no fold; ~35% bet postflop, preflop BB option raises less.
    # ctx 1 (facing a bet): ~30% fold, 55% call, 15% raise.
    # ctx 2 (facing our raise after they bet/raised this street): folds less, re-raises more.
    cnt = np.zeros((4, 3, 3))
    cnt[:, 0] = [0.0, 6.5, 3.5]
    cnt[0, 0] = [0.0, 7.0, 3.0]
    cnt[:, 1] = [3.0, 5.5, 1.5]
    cnt[0, 1] = [2.0, 6.0, 2.0]
    cnt[:, 2] = [2.5, 5.5, 2.0]
    return {
        "cnt": cnt,
        "logpi": np.tile(np.log(PRIOR_PI), (4, 3, 1)),   # per street and context
        "acts": [],          # this hand: (street, ctx, action)
        "n_sd": 0,
    }


def observe(st: dict, street: int, ctx: int, action: int) -> None:
    st["cnt"][street, ctx, action] += 1.0
    st["acts"].append((street, ctx, action))


def new_hand(st: dict) -> None:
    st["acts"] = []


def _freqs(st: dict, street: int, ctx: int, price: float | None = None) -> np.ndarray:
    c = st["cnt"][street, ctx].copy()
    if ctx == 0:
        c[FOLD] = 0.0
    elif price is not None and ctx >= 1:
        # bigger bets get more folds, fewer calls (adjustment around a 0.28 price)
        c[FOLD] *= math.exp(3.0 * (price - 0.28))
        c[CALL] *= math.exp(-1.5 * (price - 0.28))
    return c / c.sum()


def action_probs(st: dict, street: int, ctx: int, x: np.ndarray, price: float | None = None):
    """P(fold), P(call/check), P(raise) given strength x for each (beta, bluff) model.

    raise = value part sig(a_v + beta z) (mean r (1 - b)) + bluff part sig(a_b - beta z) (mean r b);
    fold (facing a bet) = sig(a_f - beta z) (mean f); call/check = the rest. Shapes (models, len(x)),
    each floored at FLOOR.
    """
    f = _freqs(st, street, ctx, price)
    z = x[None, :] - 0.5
    # an action the opponent almost always takes cannot say much about its cards
    flatten = min(1.0, 4.0 * (1.0 - float(f.max())))
    bet = BETAS * flatten
    b = bet[:, None]
    pr = f[RAISE]
    if pr > 0:
        av = np.array([_icpt(pr * (1.0 - bl), v) for v, bl in zip(bet, BLUFFS)])[:, None]
        p_raise = 1.0 / (1.0 + np.exp(-(av + b * z)))
        if BLUFFS.max() > 0:
            has_b = BLUFFS > 0
            ab = np.array([_icpt(pr * bl, v) if bl > 0 else 0.0 for v, bl in zip(bet, BLUFFS)])[:, None]
            p_raise = p_raise + np.where(has_b[:, None], 1.0 / (1.0 + np.exp(-(ab - b * z))), 0.0)
    else:
        p_raise = np.zeros((len(BETAS), len(x)))
    if ctx >= 1 and f[FOLD] > 0:
        af = np.array([_icpt(f[FOLD], v) for v in bet])[:, None]
        p_fold = 1.0 / (1.0 + np.exp(-(af - b * z)))
    else:
        p_fold = np.zeros_like(p_raise)
    tot = p_raise + p_fold
    over = tot > 1.0
    if over.any():
        p_raise = np.where(over, p_raise / tot, p_raise)
        p_fold = np.where(over, p_fold / tot, p_fold)
    p_call = 1.0 - p_raise - p_fold
    fl = lambda p: FLOOR + (1.0 - FLOOR) * np.clip(p, 0.0, 1.0)
    return fl(p_fold), fl(p_call), fl(p_raise)


def action_logp(st: dict, street: int, ctx: int, action: int, x: np.ndarray, price: float | None = None) -> np.ndarray:
    """log P(action | x, model) for each model: shape (len(BETAS), len(x))."""
    return np.log(action_probs(st, street, ctx, x, price)[action])


def _groups(st: dict) -> dict:
    g: dict = {}
    for s, ctx, a in st["acts"]:
        g.setdefault((s, ctx), []).append(a)
    return g


def log_weights(st: dict, x_by_street: dict) -> np.ndarray:
    """log w per sample from this hand's opponent actions. x_by_street[s] = strength percentiles.

    Informativeness beta is uncertain per (street, context); each group mixes over its posterior.
    """
    n = len(next(iter(x_by_street.values())))
    lw = np.zeros(n)
    for (s, ctx), acts in _groups(st).items():
        x = x_by_street.get(s)
        if x is None:
            continue
        lp = np.zeros((len(BETAS), n))
        for a in acts:
            lp += action_logp(st, s, ctx, a, x)
        lp += st["logpi"][s, ctx][:, None]
        m = lp.max(axis=0)
        lw += m + np.log(np.exp(lp - m).sum(axis=0))      # log sum_beta pi * prod P
    return lw


def learn_showdown(st: dict, x_actual: dict, decay: float = 0.985, clip: float = 0.6) -> None:
    """Update each (street, context) beta posterior from the hand just shown down.

    x_actual[s] = the opponent's real strength percentile on street s. Evidence for beta is the
    likelihood of their actions given the real strength, relative to a uniform-random strength.
    """
    grid = np.linspace(0.005, 0.995, 100)
    for (s, ctx), acts in _groups(st).items():
        if s not in x_actual:
            continue
        xa = np.array([x_actual[s]])
        lp_a = np.zeros(len(BETAS))
        lp_ref = np.zeros((len(BETAS), len(grid)))
        for a in acts:
            lp_a += action_logp(st, s, ctx, a, xa)[:, 0]
            lp_ref += action_logp(st, s, ctx, a, grid)
        ref = np.log(np.exp(lp_ref).mean(axis=1))
        ev = np.clip(lp_a - ref - (lp_a - ref).mean(), -clip, clip)
        lpi = st["logpi"][s, ctx] + ev
        pi = np.exp(lpi - lpi.max())
        pi /= pi.sum()
        pi = decay * pi + (1.0 - decay) * PRIOR_PI
        st["logpi"][s, ctx] = np.log(pi)
    st["n_sd"] += 1


def response_probs(st: dict, street: int, x: np.ndarray, price: float, can_raise: bool = True, ctx: int = 1):
    """Per-sample P(fold), P(call), P(raise) of the opponent facing our bet at this price,
    mixed over the (street, ctx) model posterior; ctx 2 = they already bet/raised this street."""
    pi = np.exp(st["logpi"][street, ctx])[:, None]
    pf, pc, pr = ((pi * p).sum(axis=0) for p in action_probs(st, street, ctx, x, price))
    if not can_raise:
        pc, pr = pc + pr, np.zeros_like(pr)
    tot = pf + pc + pr
    return pf / tot, pc / tot, pr / tot


def pi_mean_beta(st: dict) -> list:
    """Posterior mean (beta, bluff share) per street and context (diagnostics)."""
    out = []
    for s in range(4):
        row = []
        for c in (0, 1, 2):
            pi = np.exp(st["logpi"][s, c])
            row.append([round(float((pi * BETAS).sum()), 1), round(float((pi * BLUFFS).sum()), 2)])
        out.append(row)
    return out
