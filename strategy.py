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

import math
import os
import time
from math import comb

import numpy as np

import fasteval
import range_bayes as RB
try:
    import stylemodel as SM
except Exception:  # pragma: no cover
    SM = None

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


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def equity_v2(hole, board, n, q_pre, q_streets, eps_streets, rng, eps_pre=0.08, tau=0.06,
              cap_x=None, cap_depth=0.5):
    """R1/R3: equity vs a range weighted per opponent action on the street it happened.

    q_streets: {street: q} for postflop streets where the opponent raised; each raise is judged
    on that street's board (made hand blended with final strength before the river), with a
    soft edge (tau) and a per-street floor weight (eps_streets[street], learned bluff share).
    cap_x: the opponent checked this street; down-weight the top cap_x of current strength.
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
        w *= eps_pre + (1.0 - eps_pre) * _sig((pr - (1.0 - q_pre)) / tau)
    if nb >= 3 and (q_streets or cap_x is not None):
        pct_final = _pct(op_s)
        cache = {}

        def strength(s):
            if s not in cache:
                k = (0, 3, 4, 5)[s]
                if k < 5:
                    cur = fasteval.partial_scores(opp, np.ascontiguousarray(full[:, :k]))
                    cache[s] = _pct(0.6 * _pct(cur) + 0.4 * pct_final)
                else:
                    cache[s] = pct_final
            return cache[s]

        for st, q in q_streets.items():
            if q < 0.999:
                e = eps_streets.get(st, eps_pre)
                w *= e + (1.0 - e) * _sig((strength(st) - (1.0 - q)) / tau)
        if cap_x is not None and cap_x > 0.0:
            cur_st = (0, 0, 0, 1, 2, 3)[nb]
            w *= 1.0 - cap_depth * _sig((strength(cur_st) - (1.0 - cap_x)) / tau)
    eq = float((w * win).sum() / w.sum())
    return eq, raw


# ---------------------------------------------------------------------------
# Parameters (tuned offline with tools/selfplay.py)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # value raise thresholds by street (equity vs range)
    "raise_eq": [0.53, 0.60, 0.64, 0.68],   # "raise_wider": passed hold-out gates 2026-11-07
    # bet thresholds when checked to (half / pot)
    "bet_half": [0.50, 0.51, 0.52, 0.54],
    "bet_pot": [0.55, 0.60, 0.63, 0.66],
    # extra equity needed above pot odds to call
    "call_margin": [0.0, 0.03, 0.03, 0.02],
    # semi-bluff when checked to
    "bluff_freq": 0.1,
    "maniac_adj": 0.10,
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
    # ---- v22-field switches (these defaults reproduce v22 exactly) ----
    "r1": 0,              # street-of-action soft ranges (equity_v2)
    "tau": 0.06,          # softness of range edges
    "sd_k": 8.0,          # R2: showdown calibration weight n/(n + sd_k)
    "learn_eps": 0,       # R2: floor weight from the learned bluff share of shown raises
    "bluff_prior": 0.15,  # ... prior bluff share, prior strength 4 showdowns
    "r3_cap": 0.0,        # R3: depth of the check cap (0 = off)
    "agg_lo": 0.40,       # D1/D2: opponent counts as aggressive above this raise rate
    "d1_call": 0.0,       # D1: call margin reduction vs aggressive opponents
    "d2_raise": 0.0,      # D2: value re-raise threshold reduction vs aggressive opponents
    "d2_no_maniac": 0,    # D2: drop the +maniac_adj penalty
    "d3_ev": 0,           # D3: EV bluff-raise facing a bet (uses fold-to-our-re-raise)
    "d3_prior": 0.20,     # ... prior fold rate to our raise over their bet (8 pseudo-obs)
    "d3_called": 0.90,    # ... realisation of called equity before the river
    "d3_fold_k": 0.35,    # ... how much better we do vs the hands that fold
    "d3_min_obs": 0,      # ... responses seen on this street before D3 may fire
    "d3_margin": 0.05,    # ... EV margin, fraction of pot
    "d4_commit": 0,       # D4: SPR stack-off rule
    "d4_margin": 0.02,
    "bayes": 0,           # Bayesian action-likelihood range (range_bayes) instead of top-q cuts
    "ev": 0,              # with bayes: universal EV decision (ev_decide) instead of thresholds
    "rz_ip": 0.95,        # equity realisation base in / out of position
    "rz_oop": 0.85,
    "ev_margin": 0.025,   # a raise must beat the passive line by this x pot (+0.1 chips)
    "rz_k": 0.0,          # strength-dependent realisation slope (0 = off)
    "ev_adapt": 0,        # learn fold scale, called-equity bias and margin from this opponent
    "ev_margin_min": 0.02,
    "ev_margin_extra": 0.06,
    "rz_weak_only": 0,
    "classify": 0,        # learned style advisor (stylemodel) blends profile parameters
    "fold_mult": 1.0,     # exploit levers (set per style by the advisor): predicted-fold multiplier,
    "call_bias": 0.0,     # ... bonus x pot for calling, ...
    "trap": 0.0,          # ... value of checking a strong hand to a frequent bettor
    "rw": 0.0,            # raise-war calibration weight: learn (showdown outcome - model equity)
    "rw_k": 8.0,          # ... in pots where they raised over our bet/raise; pseudo-observations
    "rw_prior": 0.0,      # ... prior equity bias before evidence
    "rw_lo": -0.25,       # ... clip of the learned bias
    "rw_hi": 0.05,
    "fb": 0.0,            # facing-bet calibration weight: learn (showdown outcome - model equity)
    "fb_k": 6.0,          # ... per street where they bet into us (not a raise war); pseudo-obs
    "fb_lo": -0.20,       # ... clip of the learned bias
    "fb_hi": 0.20,
    "pc": 0.0,            # preflop-call calibration weight: learn (realised - predicted) EV of
    "pc_k": 10.0,         # ... calling their preflop raise, x pot; pseudo-observations
    "pc_lo": -0.50,       # ... clip of the learned bias (fraction of the pot after calling)
    "pc_hi": 0.20,
}

# v22-field candidate profile: Bayesian ranges; decision-rule changes (d1-d4) off until measured
FIELD_PARAMS = dict(DEFAULT_PARAMS)
FIELD_PARAMS.update({"bayes": 1, "ev": 1,                   # EV v3 (Bayesian ranges + EV decisions)
                     "rw": 1.0, "rw_k": 4.0,                # raise-war calibration
                     "classify": 0,                         # style advisor off
                     "fb": 1.0, "fb_k": 3.0,                # facing-bet calibration
                     "pc": 1.0,                             # preflop-call calibration
                     "fold_mult": 1.4, "ev_margin": 0.0})   # bluff more, raise whenever +EV


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
        "last_was_my_raise": False,
        "think": 0.0,                  # seconds spent in choose_action
        "calls": 0,
        "results": [],                 # per-hand chip results (bandit use)
        "sd_raise_low": [0, 0],        # shown raising hands below median strength [pre, post]
        "opp_checked": [0, 0, 0, 0],   # opponent checked on this street, this hand
        "my_reraise": False,           # our last raise was over their bet
        "rr_faced": [0, 0, 0, 0],      # our raises over their bet the opponent responded to
        "rr_fold": [0, 0, 0, 0],
        "rr_raise": [0, 0, 0, 0],      # ... and re-raised
        "opp_bet_opp": [0, 0, 0, 0],   # opponent had the option to bet (checked to / first) ...
        "opp_bet": [0, 0, 0, 0],       # ... and bet
        "rb": RB.new_state(),          # Bayesian range model (P["bayes"])
        "c_me": [0, 0, 0, 0],          # chips in per street this hand (for action contexts)
        "c_op": [0, 0, 0, 0],
        # online calibration of the EV model vs this opponent (P["ev_adapt"])
        "cal": {"fp": 0.0, "fa": 0.0, "n": 0, "eb": 0.0, "en": 0},
        "pend": None,                  # (hand, predicted fold prob, predicted equity if called)
        "pend_sd": None,               # (hand, predicted equity if called) awaiting showdown
        # raise wars (P["rw"]): model equity vs showdown result when we continued after their raise
        "rw_b": 0.0, "rw_n": 0, "rw_pend": None,
        # facing their bet (P["fb"]): per street [flop, turn, river] sums and counts
        "fb_b": [0.0, 0.0, 0.0], "fb_n": [0, 0, 0], "fb_pend": [],
        # calling their preflop raise (P["pc"]): sum of (realised - predicted) / pot, count
        "pc_b": 0.0, "pc_n": 0, "pc_pend": None,
    }


def _process_history(m: dict, obs) -> None:
    if obs.hand_index != m["hand"]:
        m["hand"] = obs.hand_index
        m["seen"] = 0
        m["opp_raised_streets"] = [0, 0, 0, 0]
        m["last_was_my_raise"] = False
        m["my_reraise"] = False
        m["opp_checked"] = [0, 0, 0, 0]
        m["my_hole"] = tuple(obs.hole_cards)
        m["c_me"] = [1 if obs.is_button else 2, 0, 0, 0]
        m["c_op"] = [2 if obs.is_button else 1, 0, 0, 0]
        m["op_raised_st"] = [0, 0, 0, 0]
        if "rb" in m:
            RB.new_hand(m["rb"])
    hist = obs.action_history
    for a in hist[m["seen"]:]:
        s = int(a.street)
        if "rb" in m:
            if not a.is_me and m.get("pend") and m["pend"][0] == obs.hand_index:
                _, pf_pred, eqc_pred = m["pend"]
                cal = m["cal"]
                cal["fp"] += pf_pred
                cal["n"] += 1
                if int(a.action) == CHECK_CALL:
                    m["pend_sd"] = (obs.hand_index, eqc_pred)
                m["pend"] = None
            if not a.is_me:
                ctx = (2 if m["op_raised_st"][s] else 1) if m["c_me"][s] > m["c_op"][s] else 0
                act = int(a.action)
                RB.observe(m["rb"], s, ctx, RB.RAISE if act >= RAISE_HALF else (RB.FOLD if act == FOLD else RB.CALL))
                if act >= RAISE_HALF:
                    m["op_raised_st"][s] = 1
            (m["c_me"] if a.is_me else m["c_op"])[s] = a.street_total
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
            if m.get("my_reraise"):
                m["rr_faced"][m["my_raise_street"]] += 1
                m["rr_fold"][m["my_raise_street"]] += int(int(a.action) == FOLD)
                m["rr_raise"][m["my_raise_street"]] += int(int(a.action) >= RAISE_HALF)
        elif s >= FLOP and int(a.action) != FOLD and a.street_total == a.amount:
            # nothing to call when they acted (a check, or a first bet on the street)
            if int(a.action) == CHECK_CALL and a.amount == 0:
                m["opp_bet_opp"][s] += 1
                m["opp_checked"][s] = 1
            elif int(a.action) >= RAISE_HALF and _first_bet(hist, a):
                m["opp_bet_opp"][s] += 1
                m["opp_bet"][s] += 1
        if int(a.action) >= RAISE_HALF:
            m["opp_raise"][s] += 1
            m["opp_raised_streets"][s] += 1
        m["last_was_my_raise"] = False
        m["my_reraise"] = False
    m["seen"] = len(hist)


def _first_bet(hist, a) -> bool:
    """True if `a` (an opponent raise) was the first bet of its street."""
    for b in hist:
        if b is a:
            return True
        if b.street == a.street and int(b.action) >= RAISE_HALF:
            return False
    return True


def on_result(m: dict, result) -> None:
    """Update the model from a finished hand."""
    if result.hand_index != m["hand"] and "rb" in m and not result.went_to_showdown and result.my_net > 0:
        # We never acted and won: they folded their small blind facing our big blind.
        RB.observe(m["rb"], PREFLOP, 1, RB.FOLD)
    if result.hand_index == m["hand"]:
        # The opponent's last action before a showdown is never shown to us: they called our final
        # raise, or (river) checked behind our check. Record it so frequencies match the real game.
        la = m.get("last_act")
        if "rb" in m and result.went_to_showdown and la and la[0] == result.hand_index:
            _, st_, a_, tc_, btn_ = la
            if a_ >= RAISE_HALF:
                RB.observe(m["rb"], st_, 2 if m.get("op_raised_st", [0] * 4)[st_] else 1, RB.CALL)
            elif a_ == CHECK_CALL and tc_ == 0 and st_ == RIVER and not btn_:
                RB.observe(m["rb"], RIVER, 0, RB.CALL)
        # Opponent folded to our last raise (not visible in our history).
        rs = m["my_raise_street"]
        if not result.went_to_showdown and result.my_net > 0 and m["last_was_my_raise"]:
            m["faced"] += 1
            m["fold_to_raise"] += 1
            m["faced_s"][rs] += 1
            m["fold_s"][rs] += 1
            if "rb" in m:
                RB.observe(m["rb"], rs, 2 if m.get("op_raised_st", [0] * 4)[rs] else 1, RB.FOLD)
            if m.get("my_reraise"):
                m["rr_faced"][rs] += 1
                m["rr_fold"][rs] += 1
        elif m["last_was_my_raise"] and result.went_to_showdown:
            m["faced"] += 1  # they called our final raise
            m["faced_s"][rs] += 1
            if m.get("my_reraise"):
                m["rr_faced"][rs] += 1
        if (result.went_to_showdown and result.opp_hole is not None and "rb" in m and len(result.board) == 5
                and not m.get("cheap")):
            try:
                RB.learn_showdown(m["rb"], _actual_strength(m["rb"], result, m.get("my_hole")))
            except Exception:
                pass
        if result.went_to_showdown and result.opp_hole is not None and not m.get("cheap"):
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
                pct = float(((theirs > sc) + 0.5 * (theirs == sc)).mean())
                m["sd_raise_n"][1] += 1
                m["sd_raise_pct"][1] += pct
                m["sd_raise_low"][1] += int(pct < 0.5)
    rp = m.get("rw_pend")
    if rp and rp[0] == result.hand_index and result.went_to_showdown:
        outcome = 1.0 if result.my_net > 0 else (0.0 if result.my_net < 0 else 0.5)
        m["rw_b"] = m.get("rw_b", 0.0) + outcome - rp[1]
        m["rw_n"] = m.get("rw_n", 0) + 1
    m["rw_pend"] = None
    pp = m.get("pc_pend")
    if pp and pp[0] == result.hand_index:
        _, pred, sunk, pot_c = pp
        realised = result.my_net + sunk           # calling vs folding then (losing what was in)
        m["pc_b"] = m.get("pc_b", 0.0) + min(1.5, max(-1.5, (realised - pred) / max(1.0, pot_c)))
        m["pc_n"] = m.get("pc_n", 0) + 1
    m["pc_pend"] = None
    fp = m.get("fb_pend")
    if fp:
        if result.went_to_showdown and fp[0][0] == result.hand_index:
            outcome = 1.0 if result.my_net > 0 else (0.0 if result.my_net < 0 else 0.5)
            fb_b, fb_n = m.setdefault("fb_b", [0.0] * 3), m.setdefault("fb_n", [0] * 3)
            for _, i, e in fp:
                fb_b[i] += outcome - e
                fb_n[i] += 1
        fp.clear()
    if "cal" in m:
        cal = m["cal"]
        pend = m.get("pend")
        if pend and pend[0] == result.hand_index:
            cal["fp"] += pend[1]
            cal["n"] += 1
            if not result.went_to_showdown and result.my_net > 0:
                cal["fa"] += 1.0                     # they folded to our raise
            elif result.went_to_showdown:
                m["pend_sd"] = (result.hand_index, pend[2])   # called on the river
        sd = m.get("pend_sd")
        if sd and sd[0] == result.hand_index and result.went_to_showdown:
            outcome = 1.0 if result.my_net > 0 else (0.0 if result.my_net < 0 else 0.5)
            cal["eb"] += outcome - sd[1]
            cal["en"] += 1
        m["pend"] = None
        m["pend_sd"] = None
    m["results"].append(result.my_net)
    m["last_was_my_raise"] = False
    m["my_reraise"] = False


def _street_x(opp: np.ndarray, full: np.ndarray, s: int, pct_final: np.ndarray) -> np.ndarray:
    """Strength percentile of each row's hand on street s (same yardstick for samples and showdowns)."""
    if s == PREFLOP:
        return PF_RANK[colex_rows(opp)].astype(np.float64)
    if s == RIVER:
        return pct_final
    k = 3 if s == FLOP else 4
    cur = fasteval.partial_scores(opp, np.ascontiguousarray(full[:, :k]))
    return _pct(0.6 * _pct(cur) + 0.4 * pct_final)


def _actual_strength(st: dict, result, my_hole=()) -> dict:
    """The shown opponent hand's percentile on each street it acted, vs random hands."""
    streets = sorted({s for s, _, _ in st["acts"]})
    opp_hole = list(result.opp_hole)
    board = list(result.board)
    known = set(board) | set(opp_hole) | set(my_hole or ())
    rest = np.array([c for c in range(36) if c not in known], dtype=np.intp)
    n = 100
    rng = np.random.default_rng(result.hand_index + 17)
    others = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :5]
    hands = np.vstack([np.array([opp_hole], dtype=np.intp), others])
    full = np.broadcast_to(np.array(board, dtype=np.intp), (n + 1, 5))
    pct_final = _pct(fasteval.omaha_scores(hands, full))
    out = {}
    for s in streets:
        if s == PREFLOP:
            out[s] = float(preflop_rank(opp_hole))
        else:
            out[s] = float(_street_x(hands, full, s, pct_final)[0])
    return out


def bayes_samples(hole, board, n, st, rng, street):
    """Sampled opponent hands with Bayesian weights: win share, weight, current-street strength."""
    known = set(hole) | set(board)
    rest = np.array([c for c in range(36) if c not in known], dtype=np.intp)
    nb = len(board)
    draw = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :5 + (5 - nb)]
    opp = draw[:, :5]
    if nb:
        full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.intp), (n, nb)), draw[:, 5:]], axis=1)
    else:
        full = draw[:, 5:]
    me = np.broadcast_to(np.array(hole, dtype=np.intp), (n, 5))
    my_s = fasteval.omaha_scores(me[:1], full[:1])[0] if nb == 5 else fasteval.omaha_scores(me, full)
    op_s = fasteval.omaha_scores(opp, full)
    win = (my_s > op_s) + 0.5 * (my_s == op_s)
    pct_final = _pct(op_s)
    streets = sorted({s for s, _, _ in st["acts"]} | {street})
    xs = {s: _street_x(opp, full, s, pct_final) for s in streets if s == PREFLOP or (0, 3, 4, 5)[s] <= nb}
    if st["acts"]:
        lw = RB.log_weights(st, xs)
        w = np.exp(lw - lw.max())
    else:
        w = np.ones(n)
    return {"win": win, "w": w, "x": xs[street], "raw": float(win.mean()),
            "eq": float((w * win).sum() / w.sum())}


def equity_bayes(hole, board, n, st, rng):
    """Equity vs the opponent range weighted by the Bayesian action model (range_bayes)."""
    street = {0: PREFLOP, 3: FLOP, 4: TURN, 5: RIVER}[len(board)]
    S = bayes_samples(hole, board, n, st, rng, street)
    return S["eq"], S["raw"]


def _rho(P: dict, street: int, spr: float, ip: bool, eq: float = 0.5) -> float:
    """Equity realisation: 1 on the river or all-in; less with chips behind, more so out of position.

    With rz_k > 0 it also depends on the hand: with chips still to play, strong hands
    realise more than their equity (paid off later) and weak hands less (they fold to later bets).
    """
    if street >= RIVER or spr <= 0:
        return 1.0
    base = P["rz_ip"] if ip else P["rz_oop"]
    rho = 1.0 - (1.0 - base) * spr / (spr + 1.0)
    k = P.get("rz_k", 0.0)
    if k:
        d = eq - 0.5
        if P.get("rz_weak_only"):
            d = min(0.0, d)       # only weak hands realise less; leave value hands' lines alone
        rho *= min(1.5, max(0.5, 1.0 + k * d * min(1.0, spr)))
    return rho


EV_DEBUG = False   # offline replay: keep the last EV table in the model dict


def ev_decide(obs, m: dict, P: dict, S: dict, street: int):
    """Universal EV choice from the Bayesian range: fold / check-call / raise half / raise pot.

    Opponent responses to our raises (fold / call / re-raise) come from the same action model,
    per sampled hand, at the price we offer; our equity when called or re-raised is the equity
    vs that reweighted sub-range. Every amount is capped by the chips actually left.
    """
    legal = [int(a) for a in obs.legal_actions]
    st = m["rb"]
    w, win, x = S["w"], S["win"], S["x"]
    W = w.sum()
    eq = float((w * win).sum() / W)
    pot = obs.pot
    call = min(obs.to_call, obs.my_stack)
    ip = bool(obs.is_button) if street >= FLOP else not obs.is_button
    # raise war: they raised over our bet/raise on this street. The model's equity there is
    # checked against this opponent's showdowns and shifted by the learned bias.
    war = street >= FLOP and call > 0 and m.get("c_me", [0] * 4)[street] > 0
    rwb = 0.0
    if P.get("rw") and street >= FLOP:
        k = P["rw_k"]
        rwb = P["rw"] * min(P["rw_hi"], max(P["rw_lo"], (m.get("rw_b", 0.0) + P["rw_prior"] * k) / (m.get("rw_n", 0) + k)))
    eq_model = eq
    if war:
        eq = min(1.0, max(0.0, eq + rwb))
    # facing their bet (not a raise war): the model's equity is checked against this opponent's
    # showdowns per street, shrunk to the pooled bias, then to 0
    facing = street >= FLOP and call > 0 and not war
    fbb = 0.0
    if facing and P.get("fb"):
        fb_b, fb_n, k = m.get("fb_b", [0.0] * 3), m.get("fb_n", [0] * 3), P["fb_k"]
        pooled = sum(fb_b) / (sum(fb_n) + k)
        i = street - FLOP
        fbb = P["fb"] * min(P["fb_hi"], max(P["fb_lo"], (fb_b[i] + k * pooled) / (fb_n[i] + k)))
        eq = min(1.0, max(0.0, eq + fbb))
    evs = {}
    if call > 0:
        pot_c = pot + call
        left = min(obs.my_stack - call, obs.opp_stack)
        evs[FOLD] = 0.0
        evs[CHECK_CALL] = _rho(P, street, left / pot_c, ip, eq) * eq * pot_c - call
        evs[CHECK_CALL] += P.get("call_bias", 0.0) * pot          # exploit: catch an aggressor's air
        pc_spot = street == PREFLOP and m.get("op_raised_st", [0] * 4)[PREFLOP]
        pc_pred = evs[CHECK_CALL]
        if pc_spot and P.get("pc"):
            # what calling their preflop raise has really earned vs this opponent (post-flop play
            # included), relative to what the model predicted
            k = P["pc_k"]
            pcb = min(P["pc_hi"], max(P["pc_lo"], m.get("pc_b", 0.0) / (m.get("pc_n", 0) + k)))
            evs[CHECK_CALL] += P["pc"] * pcb * pot_c
    else:
        evs[CHECK_CALL] = _rho(P, street, min(obs.my_stack, obs.opp_stack) / max(1, pot), ip, eq) * eq * pot
        trap = P.get("trap", 0.0)
        if trap and street >= FLOP:
            # exploit: a strong hand checked to a frequent bettor collects their bet
            bet_rate = float(RB._freqs(st, street, 0)[RB.RAISE])
            evs[CHECK_CALL] += trap * bet_rate * max(0.0, 2.0 * eq - 1.0) * pot
    fmult = P.get("fold_mult", 1.0)
    best_passive = max(evs.values())
    best = max(evs, key=evs.get)
    cal = m.get("cal")
    adapt = bool(P.get("ev_adapt")) and cal is not None
    if adapt:
        # margin shrinks as this opponent confirms the model (cautious early, tighter later)
        mfrac = P["ev_margin_min"] + P["ev_margin_extra"] * math.sqrt(30.0 / (30.0 + cal["n"]))
        fscale = min(1.4, max(0.4, (cal["fa"] + 7.0) / (cal["fp"] + 7.0)))
        ebias = min(0.15, max(-0.15, cal["eb"] / (cal["en"] + 30.0)))
    else:
        mfrac, fscale, ebias = P["ev_margin"], 1.0, 0.0
    margin = mfrac * pot + 0.1 + (0.25 if street == PREFLOP else 0.0)
    preds = {}
    cap = 4
    for a in (RAISE_HALF, RAISE_POT):
        if a not in legal:
            continue
        to = obs.raise_to.get(a)
        if to is None:
            continue
        invest = to - obs.my_bet
        d = max(0, min(obs.opp_stack, to - obs.opp_bet))
        pot_c = pot + invest + d
        my_left, opp_left = obs.my_stack - invest, obs.opp_stack - d
        can_rr = min(my_left, opp_left) > 0 and obs.raises_this_street + 1 < cap
        rctx = 2 if m.get("op_raised_st", [0] * 4)[street] else 1
        pf, pc, pr = RB.response_probs(st, street, x, d / max(1, pot_c), can_rr, rctx)
        Pf, Pc, Pr = float((w * pf).sum() / W), float((w * pc).sum() / W), float((w * pr).sum() / W)
        wc, wr = w * pc, w * pr
        eq_c = float((wc * win).sum() / max(1e-12, wc.sum())) if wc.sum() > 1e-9 else eq
        if war:
            eq_c = min(1.0, max(0.0, eq_c + rwb))
        elif fbb:
            eq_c = min(1.0, max(0.0, eq_c + fbb))
        preds[a] = (Pf, eq_c)
        if fmult != 1.0:                       # exploit: nits fold more than even the model expects
            pf_new = min(0.95, Pf * fmult)
            Pc = max(0.0, Pc - (pf_new - Pf))
            Pf = pf_new
        if adapt:
            pf_new = min(0.95, Pf * fscale)
            Pc = max(0.0, Pc - (pf_new - Pf))
            Pf = pf_new
            eq_c = min(1.0, max(0.0, eq_c + ebias))
        eq_r = float((wr * win).sum() / max(1e-12, wr.sum())) if wr.sum() > 1e-9 else eq
        eq_r = min(1.0, max(0.0, eq_r + rwb))          # their re-raise starts a raise war
        behind = min(my_left, opp_left)
        ev_called = _rho(P, street, behind / pot_c, ip, eq_c) * eq_c * pot_c - invest
        if can_rr:
            r_inc = min(pot_c, behind)
            pot_rr = pot_c + 2 * r_inc
            ev_rr = max(-float(invest), _rho(P, street, (behind - r_inc) / pot_rr, ip, eq_r) * eq_r * pot_rr - (invest + r_inc))
        else:
            ev_rr = ev_called
        evs[a] = Pf * pot + Pc * ev_called + Pr * ev_rr
        if evs[a] > max(best_passive + margin, evs.get(best, -1e9) if best in (RAISE_HALF, RAISE_POT) else -1e9):
            best = a
    if best == FOLD and FOLD not in legal:
        best = CHECK_CALL
    if best in preds:
        m["pend"] = (obs.hand_index, preds[best][0], preds[best][1])   # raw model predictions
    if war and best != FOLD:
        m["rw_pend"] = (obs.hand_index, eq_model)
    if call > 0 and street == PREFLOP and m.get("op_raised_st", [0] * 4)[PREFLOP] and best == CHECK_CALL:
        m["pc_pend"] = (obs.hand_index, pc_pred, obs.my_bet, pot + call)
    if facing and best != FOLD:
        pend = m.setdefault("fb_pend", [])
        if pend and pend[0][0] != obs.hand_index:
            pend.clear()
        pend.append((obs.hand_index, street - FLOP, eq_model))
    if EV_DEBUG:
        m["ev_dbg"] = dict(evs=dict(evs), preds=dict(preds), eq=eq, margin=margin)
    return best


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
        wsd = nsd / (nsd + P.get("sd_k", 8.0))
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
    if P.get("classify") and SM is not None:
        adv = m.get("adv")
        if adv is None or adv.default is not P:
            adv = m["adv"] = SM.Advisor(P)
        P = adv.params(m, len(m["results"]))
    act = _decide(obs, m, P, n_override)
    # remember our own action now: if it ends the hand we never see it in history
    m["last_was_my_raise"] = act >= RAISE_HALF
    m["my_raise_street"] = int(obs.street)
    m["my_reraise"] = act >= RAISE_HALF and obs.to_call > 0
    m["last_act"] = (obs.hand_index, int(obs.street), int(act), int(obs.to_call), bool(obs.is_button))
    return act


def _post_agg(m: dict, P: dict) -> float:
    """Opponent's postflop raise rate (prior-smoothed)."""
    pn = P["prior_n"]
    return (sum(m["opp_raise"][1:]) + P["prior_raise"] * pn) / (sum(m["opp_act"][1:]) + pn)


def _street_ranges(m: dict, P: dict, street: int):
    """R1/R2: per-street range width q and floor weight eps for this hand's postflop raises."""
    ors = m["opp_raised_streets"]
    qs, es = {}, {}
    nsd = m["sd_raise_n"][1]
    b = None
    if P["learn_eps"]:
        b = (m["sd_raise_low"][1] + P["bluff_prior"] * 4.0) / (nsd + 4.0)
    for s in (FLOP, TURN, RIVER):
        if ors[s] and s <= street:
            q = max(P["q_min"], _raise_q(m, s, P) ** (1.0 + 0.5 * (ors[s] - 1)))
            qs[s] = q
            if b is not None and q < 0.999:
                # weight of a bottom hand that makes the raising range a b-share bluffs
                es[s] = float(min(0.8, max(P["eps"], b * q / max(1e-6, (1.0 - b) * (1.0 - q)))))
    return qs, es


def _cap_x(m: dict, P: dict, street: int):
    """R3: when the opponent checked this street, the share of top hands they would have bet."""
    if not P["r3_cap"] or street < FLOP or not m["opp_checked"][street]:
        return None
    k = 8.0
    return (m["opp_bet"][street] + 0.45 * k) / (m["opp_bet_opp"][street] + k)


def _commit(obs, eq: float, P: dict):
    """D4: at low stack-to-pot, stack off when equity beats the all-in price."""
    call = obs.to_call
    eff = min(obs.my_stack, obs.opp_stack + call)
    if eff <= 0:
        return None
    spr_after = (obs.my_stack - call) / max(1.0, obs.pot + call)
    if spr_after > 1.0:
        return None
    need_ai = eff / max(1.0, obs.pot + 2 * eff - call)
    if eq >= need_ai + P["d4_margin"]:
        return RAISE_POT if obs.my_stack > call else CHECK_CALL
    return None


def _ev_reraise_better(obs, m: dict, P: dict, eq: float, street: int, margin: float) -> bool:
    """D3: raise their bet when fold equity x pot beats calling (or folding).

    Equity when called is vs the hands that continue: the hands that fold are ones we beat, so
    e_called = (e - f * e_fold) / (1 - f) with e_fold = e + k (1 - e) (a fold-equity estimate), times a
    realisation factor before the river. The fold rate f is our raise-over-their-bet history,
    shrunk to a cautious prior until we have seen enough responses.
    """
    seen = m["rr_faced"][street]
    if seen < P["d3_min_obs"]:
        return False
    k = 8.0
    f = (m["rr_fold"][street] + P["d3_prior"] * k) / (seen + k)
    f = min(f, 0.9)
    r = min(1.0 - f, (m["rr_raise"][street] + 0.1 * k) / (seen + k))   # they re-raise: we give up
    e_fold = eq + P["d3_fold_k"] * (1.0 - eq)
    e_called = max(0.0, (eq - f * e_fold) / max(1e-6, 1.0 - f))
    if street < RIVER:
        e_called *= P["d3_called"]
    pot, call = obs.pot, obs.to_call
    raise_to = obs.raise_to.get(RAISE_POT)
    if raise_to is None:
        return False
    invest = raise_to - obs.my_bet
    opp_call = min(obs.opp_stack, raise_to - obs.opp_bet)
    final = pot + invest + opp_call
    ev_raise = f * pot - r * invest + (1.0 - f - r) * (e_called * final - invest)
    ev_call = (eq - margin) * (pot + call) - call
    return ev_raise > max(0.0, ev_call) + P["d3_margin"] * pot


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
    q_post = max(P["q_min"], q_post)

    n = n_override if n_override is not None else P["n_samples"]
    seed = (obs.hand_index * 7919 + len(obs.action_history) * 104729 + sum(hole) * 31 + street) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    if n <= 0:
        # cheap fallback: preflop table / raw equity
        eq = preflop_equity(hole) if street == PREFLOP else 0.5
    elif P.get("bayes") and P.get("ev"):
        S = bayes_samples(hole, board, n, m["rb"], rng, street)
        return ev_decide(obs, m, P, S, street)
    elif P.get("bayes"):
        eq, _raw = equity_bayes(hole, board, n, m["rb"], rng)
    elif P.get("r1"):
        eq, _raw = equity_v2(hole, board, n, q_pre, *_street_ranges(m, P, street), rng,
                             eps_pre=P["eps"], tau=P["tau"], cap_x=_cap_x(m, P, street),
                             cap_depth=P["r3_cap"])
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
        r_eq = P["raise_eq"][street] + (P["maniac_adj"] if agg > P["maniac_agg"] and not P["d2_no_maniac"] else 0.0)
        margin = P["call_margin"][street]
        if street >= FLOP and _post_agg(m, P) > P["agg_lo"]:
            r_eq -= P["d2_raise"]
            margin -= P["d1_call"]
        if P["d4_commit"]:
            c = _commit(obs, eq, P)
            if c is not None and (c != RAISE_POT or can_raise):
                return c
        if can_raise and eq >= r_eq:
            return RAISE_POT
        if P["d3_ev"] and can_raise and street >= FLOP:
            if _ev_reraise_better(obs, m, P, eq, street, margin):
                return RAISE_POT
        if eq >= price + margin:
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
        if eq >= P["bet_pot"][street]:
            return RAISE_POT
        if eq >= P["bet_half"][street]:
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
