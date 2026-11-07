"""
fasteval.py - vectorised (numpy) Short Deck Omaha evaluator.

Produces EXACTLY the same scores as evaluator.omaha_score (verified by
tools/verify_fasteval.py), but scores many (hole, board) pairs at once.

Idea: a non-flush 5-card score depends only on the multiset of ranks.
Encode a multiset as sum(5**rank) (each rank appears <= 4 times, so this is
a unique base-5 number < 5**9). Keys are additive, so the 10 hole pairs and
10 board triples combine with one broadcast add. Flushes need both hole
cards and all three board cards in one suit; then ranks are distinct and we
read the flush table instead.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import _score_ranks  # reference scoring logic (same rules)

_SIZE = 5 ** 9
NF = np.zeros(_SIZE, dtype=np.int32)   # non-flush score by rank-multiset key
FL = np.zeros(_SIZE, dtype=np.int32)   # flush / straight-flush score (distinct ranks only)

for _ranks in combinations_with_replacement(range(9), 5):
    if max(_ranks.count(r) for r in set(_ranks)) > 4:
        continue
    _key = sum(5 ** r for r in _ranks)
    _desc = tuple(sorted(_ranks, reverse=True))
    NF[_key] = _score_ranks(_desc, False)
    if len(set(_ranks)) == 5:
        FL[_key] = _score_ranks(_desc, True)

CARD_KEY = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
CARD_SUIT = np.array([c & 3 for c in range(36)], dtype=np.int8)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.intp)      # (10, 2)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.intp)    # (10, 3)
TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.intp)   # (4, 3)


def _pair_parts(hole: np.ndarray):
    """hole (N,5) card ints -> pair keys (N,10), pair flush-suit (N,10) (-1 if offsuit)."""
    k = CARD_KEY[hole]
    s = CARD_SUIT[hole].astype(np.int16)
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    pk = k[:, a] + k[:, b]
    sa, sb = s[:, a], s[:, b]
    ps = np.where(sa == sb, sa, -1)
    return pk, ps


def _triple_parts(board: np.ndarray, triples=TRIPLES):
    """board (N,5) -> triple keys (N,10), triple flush-suit (N,10) (-2 if not monotone)."""
    k = CARD_KEY[board]
    s = CARD_SUIT[board].astype(np.int16)
    a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    tk = k[:, a] + k[:, b] + k[:, c]
    sa, sb, sc = s[:, a], s[:, b], s[:, c]
    ts = np.where((sa == sb) & (sb == sc), sa, -2)
    return tk, ts


def combine(pk, ps, tk, ts) -> np.ndarray:
    """Best score per row from pair parts (N,P) and triple parts (N,T)."""
    key = pk[:, :, None] + tk[:, None, :]
    flush = ps[:, :, None] == ts[:, None, :]
    sc = np.where(flush, FL[key], NF[key])
    return sc.reshape(sc.shape[0], -1).max(axis=1)


def omaha_scores(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Vectorised omaha_score: hole (N,5), board (N,5) -> scores (N,)."""
    pk, ps = _pair_parts(hole)
    tk, ts = _triple_parts(board)
    return combine(pk, ps, tk, ts)


def partial_scores(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Best CURRENT made hand with 3 or 4 board cards (2 hole + 3 board).
    hole (N,5), board (N,3) or (N,4). Used for opponent range estimates."""
    pk, ps = _pair_parts(hole)
    nb = board.shape[1]
    if nb == 3:
        tri = np.array([[0, 1, 2]], dtype=np.intp)
    elif nb == 4:
        tri = TRIPLES4
    else:
        tri = TRIPLES
    tk, ts = _triple_parts(board, tri)
    return combine(pk, ps, tk, ts)
