"""
strategy.py - decision logic for the Short Deck PLO5 bot.

Pieces:
  * preflop lookup (offline Monte Carlo, data/preflop_*.npy)
  * range-weighted Monte Carlo equity (vectorised, fasteval)
  * opponent model (raise frequency, fold-to-raise, showdown calibration)
  * pot-odds / value / bluff decision rule with tunable parameters

agent.py wraps everything here in try/except, so any bug falls back to a
legal action. Nothing in this file may write files or print per-decision.
"""

from __future__ import annotations

import os
import time
from math import comb

import numpy as np

import fasteval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Preflop tables
# ---------------------------------------------------------------------------
PF_EQ = np.load(os.path.join(HERE, "data", "preflop_eq.npy")).astype(np.float32) / 65535.0
PF_RANK = np.load(os.path.join(HERE, "data", "preflop_rank.npy")).astype(np.float32) / 65535.0
COMB = np.array([[comb(c, k) for k in range(6)] for c in range(36)], dtype=np.int64)

# Action ids (same values as game.Action, kept as ints to avoid import cycles)
FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT = 0, 1, 2, 3
PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3


def colex_index(cards) -> int:
    s = sorted(cards)
    return int(COMB[s[0], 1] + COMB[s[1], 2] + COMB[s[2], 3] + COMB[s[3], 4] + COMB[s[4], 5])


def colex_rows(hands: np.ndarray) -> np.ndarray:
    s = np.sort(hands, axis=1)
    return (COMB[s[:, 0], 1] + COMB[s[:, 1], 2] + COMB[s[:, 2], 3]
            + COMB[s[:, 3], 4] + COMB[s[:, 4], 5])


def preflop_equity(hole) -> float:
    return float(PF_EQ[colex_index(hole)])


def preflop_rank(hole) -> float:
    return float(PF_RANK[colex_index(hole)])


# ---------------------------------------------------------------------------
# Range-weighted Monte Carlo equity
# ---------------------------------------------------------------------------

def _pct(x: np.ndarray) -> np.ndarray:
    """Percentile rank (0..1) of each element within x."""
    n = x.shape[0]
    r = np.empty(n, dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(n)
    return r / max(n - 1, 1)


def equity(hole, board, n, q_pre=1.0, q_post=1.0, rng=None, eps=0.08):
    """Return (equity vs weighted range, equity vs random hand).

    q_pre: opponent's preflop range = top q_pre fraction by preflop rank.
    q_post: opponent's postflop range = top q_post fraction (within samples)
            of a blend of current made hand and final hand strength.
    Hands outside the range keep weight eps (bluffs / misreads).
    """
    if rng is None:
        rng = np.random.default_rng()
    known = set(hole) | set(board)
    rest = np.array([c for c in range(36) if c not in known], dtype=np.intp)
    nb = len(board)
    need = 5 + (5 - nb)
    draw = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :need]
    opp = draw[:, :5]
    if nb:
        full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.intp), (n, nb)), draw[:, 5:]], axis=1)
    else:
        full = draw[:, 5:]
    me = np.broadcast_to(np.array(hole, dtype=np.intp), (n, 5))
    if nb == 5:
        my_s = fasteval.omaha_scores(me[:1], full[:1])[0]
    else:
        my_s = fasteval.omaha_scores(me, full)
    op_s = fasteval.omaha_scores(opp, full)
    win = (my_s > op_s) + 0.5 * (my_s == op_s)
    raw = float(win.mean())

    w = np.ones(n)
    if q_pre < 0.999:
        pr = PF_RANK[colex_rows(opp)]
        w *= np.where(pr >= 1.0 - q_pre, 1.0, eps)
    if q_post < 0.999 and nb >= 3:
        if nb < 5:
            cur = fasteval.partial_scores(opp, np.broadcast_to(np.array(board, dtype=np.intp), (n, nb)))
            strength = 0.6 * _pct(cur) + 0.4 * _pct(op_s)
        else:
            strength = _pct(op_s)
        w *= np.where(_pct(strength) >= 1.0 - q_post, 1.0, eps)
    eq = float((w * win).sum() / w.sum())
    return eq, raw


# ---------------------------------------------------------------------------
# Parameters (tuned offline with tools/selfplay.py)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # value raise thresholds by street (equity vs range)
    "raise_eq": [0.60, 0.68, 0.72, 0.75],
    # bet thresholds when checked to (half / pot)
    "bet_half": [0.56, 0.58, 0.60, 0.62],
    "bet_pot": [0.62, 0.68, 0.72, 0.76],
    # extra equity needed above pot odds to call
    "call_margin": [0.0, 0.03, 0.03, 0.02],
    # semi-bluff when checked to
    "bluff_freq": 0.25,
    "bluff_lo": 0.30,
    "bluff_hi": 0.52,
    # opponent model
    "q_min": 0.12,
    "prior_n": 12.0,
    "prior_raise": 0.30,
    "prior_fold": 0.35,
    "eps": 0.08,
    # samples
    "n_samples": 700,
}


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

def new_model() -> dict:
    return {
        "opp_act": [0, 0, 0, 0],       # opp actions observed per street
        "opp_raise": [0, 0, 0, 0],     # opp raises per street
        "faced": 0,                    # times opp responded to our raise
        "fold_to_raise": 0,            # ...and folded
        "sd_raise_n": [0, 0],          # showdowns where opp raised [pre, post]
        "sd_raise_pct": [0.0, 0.0],    # sum of their strength percentile
        "hand": -1,                    # current hand index
        "seen": 0,                     # history entries already processed
        "opp_raised_streets": [0, 0, 0, 0],
        "last_was_my_raise": False,
        "think": 0.0,                  # seconds spent in choose_action
        "calls": 0,
        "results": [],                 # per-hand chip results (bandit use)
    }


def _process_history(m: dict, obs) -> None:
    if obs.hand_index != m["hand"]:
        m["hand"] = obs.hand_index
        m["seen"] = 0
        m["opp_raised_streets"] = [0, 0, 0, 0]
        m["last_was_my_raise"] = False
        m["my_hole"] = tuple(obs.hole_cards)
    hist = obs.action_history
    for a in hist[m["seen"]:]:
        s = int(a.street)
        if a.is_me:
            m["last_was_my_raise"] = int(a.action) >= RAISE_HALF
            continue
        m["opp_act"][s] += 1
        if m["last_was_my_raise"]:
            m["faced"] += 1
            if int(a.action) == FOLD:
                m["fold_to_raise"] += 1
        if int(a.action) >= RAISE_HALF:
            m["opp_raise"][s] += 1
            m["opp_raised_streets"][s] += 1
        m["last_was_my_raise"] = False
    m["seen"] = len(hist)


def on_result(m: dict, result) -> None:
    """Update the model from a finished hand."""
    if result.hand_index == m["hand"]:
        # Opponent folded to our last raise (not visible in our history).
        if not result.went_to_showdown and result.my_net > 0 and m["last_was_my_raise"]:
            m["faced"] += 1
            m["fold_to_raise"] += 1
        elif m["last_was_my_raise"] and result.went_to_showdown:
            m["faced"] += 1  # they called our final raise
        if result.went_to_showdown and result.opp_hole is not None:
            ors = m["opp_raised_streets"]
            if ors[0]:
                m["sd_raise_n"][0] += 1
                m["sd_raise_pct"][0] += preflop_rank(result.opp_hole)
            if any(ors[1:]) and len(result.board) == 5:
                # strength of their final hand vs random hands on this board
                rng = np.random.default_rng(result.hand_index)
                known = set(result.board) | set(result.opp_hole)
                rest = np.array([c for c in range(36) if c not in known], dtype=np.intp)
                n = 150
                others = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :5]
                b = np.broadcast_to(np.array(result.board, dtype=np.intp), (n, 5))
                theirs = fasteval.omaha_scores(np.array([result.opp_hole], dtype=np.intp), b[:1])[0]
                sc = fasteval.omaha_scores(others, b)
                m["sd_raise_n"][1] += 1
                m["sd_raise_pct"][1] += float(((theirs > sc) + 0.5 * (theirs == sc)).mean())
    m["results"].append(result.my_net)
    m["last_was_my_raise"] = False


def _raise_q(m: dict, street: int, P: dict) -> float:
    """How narrow is the opponent's range when they raise on this street?"""
    pn, pr = P["prior_n"], P["prior_raise"]
    f = (m["opp_raise"][street] + pr * pn) / (m["opp_act"][street] + pn)
    q = f
    k = 0 if street == PREFLOP else 1
    nsd = m["sd_raise_n"][k]
    if nsd >= 3:
        mean_pct = m["sd_raise_pct"][k] / nsd
        q_sd = min(1.0, max(0.0, 2.0 * (1.0 - mean_pct)))
        wsd = nsd / (nsd + 8.0)
        q = (1 - wsd) * q + wsd * q_sd
    return min(1.0, max(P["q_min"], q))


def fold_rate(m: dict, P: dict) -> float:
    pn = P["prior_n"]
    return (m["fold_to_raise"] + P["prior_fold"] * pn) / (m["faced"] + pn)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(obs, m: dict, P: dict | None = None, n_override: int | None = None):
    """Return an action id. May raise; caller must catch."""
    if P is None:
        P = DEFAULT_PARAMS
    _process_history(m, obs)
    legal = [int(a) for a in obs.legal_actions]
    street = int(obs.street)
    hole = list(obs.hole_cards)
    board = list(obs.board)

    # opponent range narrowing from this hand's raises
    ors = m["opp_raised_streets"]
    q_pre = 1.0
    if ors[0]:
        q_pre = _raise_q(m, PREFLOP, P) ** (1.0 + 0.5 * (ors[0] - 1))
        q_pre = max(P["q_min"], q_pre)
    q_post = 1.0
    for s in (FLOP, TURN, RIVER):
        if ors[s] and s <= street:
            q_post *= _raise_q(m, s, P) ** (1.0 + 0.5 * (ors[s] - 1))
    q_post = max(P["q_min"], q_post)

    n = n_override if n_override is not None else P["n_samples"]
    seed = (obs.hand_index * 7919 + len(obs.action_history) * 104729 + sum(hole) * 31 + street) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    if n <= 0:
        # cheap fallback: preflop table / raw equity
        eq = preflop_equity(hole) if street == PREFLOP else 0.5
    else:
        eq, _raw = equity(hole, board, n, q_pre, q_post, rng, P["eps"])

    to_call = obs.to_call
    pot = obs.pot
    can_raise = RAISE_POT in legal
    fr = fold_rate(m, P)
    u = rng.random()

    if to_call > 0:
        price = to_call / (pot + to_call)
        if can_raise and eq >= P["raise_eq"][street]:
            return RAISE_POT
        if eq >= price + P["call_margin"][street]:
            return CHECK_CALL
        # occasional bluff-raise against folders (postflop only)
        if can_raise and street >= FLOP and fr > 0.5 and u < (fr - 0.5) * 0.6 and eq > 0.25:
            return RAISE_POT
        return FOLD if FOLD in legal else CHECK_CALL

    # checked to us / free option
    if can_raise:
        if eq >= P["bet_pot"][street]:
            return RAISE_POT
        if eq >= P["bet_half"][street]:
            return RAISE_HALF
        if street >= FLOP and P["bluff_lo"] <= eq <= P["bluff_hi"]:
            bf = P["bluff_freq"] * min(2.0, fr / 0.35)
            if u < bf:
                return RAISE_HALF if fr < 0.45 else RAISE_POT
    return CHECK_CALL


def _warmup() -> None:
    """Run each code path once at import so the first real decision is fast."""
    rng = np.random.default_rng(0)
    for b in ([], [0, 5, 10], [0, 5, 10, 15], [0, 5, 10, 15, 20]):
        equity([30, 31, 32, 33, 34], b, 64, 0.5, 0.5, rng)



