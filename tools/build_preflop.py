"""Offline planning: preflop equity of every 5-card short-deck hand vs a random hand.

Writes data/preflop_pct.npy: uint16 array indexed by the colex index of the
sorted 5 cards (sum C(c_i, i+1)). Value = equity * 65535.
Also data/preflop_rank.npy: percentile rank (0..65535) of each hand among all hands.
Hands are grouped by suit isomorphism so equivalent hands share one estimate.
"""
import os, sys, time
from itertools import combinations, permutations
from math import comb
from multiprocessing import Pool
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np

SAMPLES = int(os.environ.get("SAMPLES", "6000"))
PERMS = list(permutations(range(4)))


def colex(cards):
    return sum(comb(c, i + 1) for i, c in enumerate(sorted(cards)))


def canon(hand):
    best = None
    for p in PERMS:
        t = tuple(sorted((c & ~3) | p[c & 3] for c in hand))
        if best is None or t < best:
            best = t
    return best


def sim(hands):
    import fasteval
    rng = np.random.default_rng(abs(hash(hands[0])) % (2**32))
    out = []
    for h in hands:
        rest = np.array([c for c in range(36) if c not in h], dtype=np.intp)
        idx = np.argsort(rng.random((SAMPLES, 31)), axis=1)[:, :10]
        draw = rest[idx]
        me = np.broadcast_to(np.array(h, dtype=np.intp), (SAMPLES, 5))
        a = fasteval.omaha_scores(me, draw[:, 5:])
        b = fasteval.omaha_scores(draw[:, :5], draw[:, 5:])
        out.append(float(((a > b) + 0.5 * (a == b)).mean()))
    return out


if __name__ == "__main__":
    t = time.time()
    hands = list(combinations(range(36), 5))
    canon_of = [canon(h) for h in hands]
    classes = sorted(set(canon_of))
    print("hands", len(hands), "classes", len(classes), "canon s", round(time.time() - t, 1), flush=True)
    chunks = [classes[i:i + 200] for i in range(0, len(classes), 200)]
    with Pool(8) as pool:
        res = pool.map(sim, chunks)
    eq = dict(zip(classes, [x for r in res for x in r]))
    table = np.zeros(comb(36, 5), dtype=np.float64)
    for h, c in zip(hands, canon_of):
        table[colex(h)] = eq[c]
    order = table.argsort(kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.save(os.path.join(HERE, "data", "preflop_eq.npy"), np.round(table * 65535).astype(np.uint16))
    np.save(os.path.join(HERE, "data", "preflop_rank.npy"),
            np.round(rank / (len(rank) - 1) * 65535).astype(np.uint16))
    print("done s", round(time.time() - t, 1), "eq min/mean/max", table.min(), table.mean(), table.max())
