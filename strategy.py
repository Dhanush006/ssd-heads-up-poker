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

HERE = os.path.dirname(os.path.abspath(__file__))

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
    "raise_eq": [0.56, 0.64, 0.68, 0.72],
    # bet thresholds when checked to (half / pot)
    "bet_half": [0.50, 0.51, 0.52, 0.54],
    "bet_pot": [0.55, 0.60, 0.63, 0.66],
    # extra equity needed above pot odds to call
    "call_margin": [0.0, 0.03, 0.03, 0.02],
    # semi-bluff when checked to
    "bluff_freq": 0.1,
    "maniac_adj": 0.10,
    "pre_reraise_add": [0.0, 0.0, 0.0, 0.0],   # preflop raise threshold add by raises so far
    "pos_call": 0.0,                           # in-position postflop call discount
    "pos_bet": 0.0,
    "call_pow": 0.0,                           # >0: narrow range after opp calls our raise                            # in-position postflop bet discount
    "maniac_agg": 0.55,
    "bluff_fr0": 0.30,     # fold rate at which bluffing starts
    "bluff_cap": 2.5,
    "reraise_bluff": 0.0,
    "steal_freq": 0.0,
    "steal_lo": 0.35,
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
        "faced_s": [0, 0, 0, 0],       # same, per street of our raise
        "fold_s": [0, 0, 0, 0],
        "my_raise_street": 0,
        "sd_raise_n": [0, 0],          # showdowns where opp raised [pre, post]
        "sd_raise_pct": [0.0, 0.0],    # sum of their strength percentile
        "hand": -1,                    # current hand index
        "seen": 0,                     # history entries already processed
        "opp_raised_streets": [0, 0, 0, 0],
        "opp_called_streets": [0, 0, 0, 0],
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
        m["opp_called_streets"] = [0, 0, 0, 0]
        m["last_was_my_raise"] = False
        m["my_hole"] = tuple(obs.hole_cards)
    hist = obs.action_history
    for a in hist[m["seen"]:]:
        s = int(a.street)
        if a.is_me:
            m["last_was_my_raise"] = int(a.action) >= RAISE_HALF
            m["my_raise_street"] = s
            continue
        m["opp_act"][s] += 1
        if m["last_was_my_raise"]:
            m["faced"] += 1
            m["faced_s"][m["my_raise_street"]] += 1
            if int(a.action) == FOLD:
                m["fold_to_raise"] += 1
                m["fold_s"][m["my_raise_street"]] += 1
        if int(a.action) >= RAISE_HALF:
            m["opp_raise"][s] += 1
            m["opp_raised_streets"][s] += 1
        elif m["last_was_my_raise"] and int(a.action) == CHECK_CALL:
            m["opp_called_streets"][s] += 1
        m["last_was_my_raise"] = False
    m["seen"] = len(hist)


def on_result(m: dict, result) -> None:
    """Update the model from a finished hand."""
    if result.hand_index == m["hand"]:
        # Opponent folded to our last raise (not visible in our history).
        rs = m["my_raise_street"]
        if not result.went_to_showdown and result.my_net > 0 and m["last_was_my_raise"]:
            m["faced"] += 1
            m["fold_to_raise"] += 1
            m["faced_s"][rs] += 1
            m["fold_s"][rs] += 1
        elif m["last_was_my_raise"] and result.went_to_showdown:
            m["faced"] += 1  # they called our final raise
            m["faced_s"][rs] += 1
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


def fold_rate(m: dict, P: dict, street: int | None = None) -> float:
    """Opponent's fold-to-our-raise rate; per street shrinks to the overall rate."""
    pn = P["prior_n"]
    overall = (m["fold_to_raise"] + P["prior_fold"] * pn) / (m["faced"] + pn)
    if street is None:
        return overall
    k = 8.0
    return (m["fold_s"][street] + overall * k) / (m["faced_s"][street] + k)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(obs, m: dict, P: dict | None = None, n_override: int | None = None):
    """Return an action id. May raise; caller must catch."""
    if P is None:
        P = DEFAULT_PARAMS
    _process_history(m, obs)
    act = _decide(obs, m, P, n_override)
    # remember our own action now: if it ends the hand we never see it in history
    m["last_was_my_raise"] = act >= RAISE_HALF
    m["my_raise_street"] = int(obs.street)
    return act


def _decide(obs, m: dict, P: dict, n_override):
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
    # opponent called our raise: their range is the part that did not fold
    if P["call_pow"] > 0:
        occ = m["opp_called_streets"]
        if occ[0] and not ors[0]:
            q_pre *= max(0.3, 1.0 - fold_rate(m, P, PREFLOP)) ** P["call_pow"]
        for s in (FLOP, TURN, RIVER):
            if occ[s] and s <= street:
                q_post *= max(0.3, 1.0 - fold_rate(m, P, s)) ** P["call_pow"]
    q_pre = max(P["q_min"], q_pre)
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
    fr = fold_rate(m, P, street)
    u = rng.random()
    # bluff intensity: 0 vs stations, grows with how often they fold to raises
    bscale = min(P["bluff_cap"], max(0.0, (fr - P["bluff_fr0"]) / 0.2))

    if to_call > 0:
        price = to_call / (pot + to_call)
        pn = P["prior_n"]
        agg = (m["opp_raise"][street] + P["prior_raise"] * pn) / (m["opp_act"][street] + pn)
        r_eq = P["raise_eq"][street] + (P["maniac_adj"] if agg > P["maniac_agg"] else 0.0)
        if street == PREFLOP:
            rc = min(3, obs.raises_this_street)
            r_eq += P["pre_reraise_add"][rc]
        if can_raise and eq >= r_eq:
            return RAISE_POT
        ip = obs.is_button and street >= FLOP
        if eq >= price + P["call_margin"][street] - (P["pos_call"] if ip else 0.0):
            return CHECK_CALL
        # occasional bluff-raise against folders (postflop only)
        if can_raise and street >= FLOP and eq > 0.25 and u < P["reraise_bluff"] * max(0.0, fr - 0.5):
            return RAISE_POT
        # preflop: SB completing the blind is cheap; steal instead vs folders
        if street == PREFLOP and can_raise and obs.raises_this_street == 0 and eq >= P["steal_lo"] \
                and u < P["steal_freq"] * bscale:
            return RAISE_HALF
        return FOLD if FOLD in legal else CHECK_CALL

    # checked to us / free option
    if can_raise:
        pa = P["pos_bet"] if (obs.is_button and street >= FLOP) else 0.0
        if eq >= P["bet_pot"][street] - pa:
            return RAISE_POT
        if eq >= P["bet_half"][street] - pa:
            return RAISE_HALF
        if street >= FLOP and eq <= P["bluff_hi"] and (eq >= P["bluff_lo"] or fr > 0.55):
            if u < P["bluff_freq"] * bscale:
                return RAISE_HALF if fr < 0.6 else RAISE_POT
        # preflop steal vs folders (first in from the small blind, or BB option)
        if street == PREFLOP and eq >= P["steal_lo"] and u < P["steal_freq"] * bscale:
            return RAISE_HALF
    return CHECK_CALL


def _warmup() -> None:
    """Run each code path once at import so the first real decision is fast."""
    rng = np.random.default_rng(0)
    for b in ([], [0, 5, 10], [0, 5, 10, 15], [0, 5, 10, 15, 20]):
        equity([30, 31, 32, 33, 34], b, 64, 0.5, 0.5, rng)


_warmup()


def _calibrate() -> float:
    """Machine speed relative to the dev laptop (1.0 = same, 2.0 = twice as slow)."""
    try:
        rng = np.random.default_rng(1)
        times = []
        for _ in range(9):
            t = time.perf_counter()
            equity([30, 31, 32, 33, 34], [0, 5, 10], 200, 0.5, 0.5, rng)
            times.append(time.perf_counter() - t)
        times.sort()
        return min(10.0, max(0.5, times[len(times) // 2] / 0.00060))
    except Exception:
        return 2.0


SPEED = _calibrate()
